"""PIL-only VLM vision-token budgeting, grid resizing, and processor measurement helpers."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_INKLING_CELL_SIZE = (40, 40)
DEFAULT_IMAGE_TOKEN_STR = "<|unused_200054|>"
RESAMPLE_FILTER_NAME = "LANCZOS"
PINNED_PILLOW_VERSION = "12.3.0"


def _resolve_lanczos_filter(image_mod: Any) -> Any:
    resampling = getattr(image_mod, "Resampling", None)
    if resampling is not None and hasattr(resampling, "LANCZOS"):
        return resampling.LANCZOS
    if hasattr(image_mod, "LANCZOS"):
        return image_mod.LANCZOS
    raise RuntimeError("Pillow LANCZOS resampling filter is unavailable")


def inspect_pillow_runtime(*, strict: bool = False) -> dict[str, str]:
    """Return active Pillow version and resample filter, failing if strict pin mismatches."""
    import PIL
    from PIL import Image

    _resolve_lanczos_filter(Image)
    ver = str(getattr(PIL, "__version__", "unknown"))
    if strict and ver != PINNED_PILLOW_VERSION:
        raise RuntimeError(
            f"Pillow version {ver!r} != pinned {PINNED_PILLOW_VERSION!r} required for byte-reproducible Arrow export"
        )
    return {"pillow_version": ver, "resample_filter": RESAMPLE_FILTER_NAME}


def read_image_cell_size(model_dir_or_cfg: Path | str | Mapping[str, Any]) -> tuple[int, int]:
    """Read pixel cell geometry ``(cell_w, cell_h)`` from processor config JSON or mapping."""
    if isinstance(model_dir_or_cfg, Mapping):
        payload: Mapping[str, Any] = model_dir_or_cfg
    else:
        base = Path(model_dir_or_cfg)
        cfg_file: Path | None = None
        if base.is_file():
            cfg_file = base
        elif base.is_dir():
            for candidate in ("processor_config.json", "preprocessor_config.json"):
                cand_path = base / candidate
                if cand_path.is_file():
                    cfg_file = cand_path
                    break
        if cfg_file is None:
            raise FileNotFoundError(
                f"Neither processor_config.json nor preprocessor_config.json found under {base}"
            )
        payload = json.loads(cfg_file.read_text(encoding="utf-8"))

    ip = payload.get("image_processor", payload)
    if not isinstance(ip, Mapping):
        raise ValueError("Invalid image_processor section in processor config")

    size = ip.get("size")
    if isinstance(size, Mapping):
        w = int(size.get("width") or 0)
        h = int(size.get("height") or 0)
        if w > 0 and h > 0:
            return (w, h)
        raise ValueError(f"Invalid positive width/height in image_processor.size: {size!r}")
    if isinstance(size, int) and size > 0:
        return (size, size)

    patch_size = int(ip.get("patch_size") or 0)
    merge_size = int(ip.get("merge_size") or 1)
    if patch_size > 0 and merge_size > 0:
        cell = patch_size * merge_size
        return (cell, cell)

    raise ValueError(f"Cannot infer image cell size from processor config keys: {sorted(ip.keys())}")


def parse_cell_size(raw: str | None, tokenizer_ref: str | None = None) -> tuple[int, int]:
    """Parse ``--cell-size`` ('W,H' or 'N') or infer cell geometry from a staged tokenizer dir."""
    if raw:
        parts = [int(x.strip()) for x in raw.lower().replace("x", ",").split(",") if x.strip()]
        if len(parts) == 1 and parts[0] > 0:
            return (parts[0], parts[0])
        if len(parts) == 2 and parts[0] > 0 and parts[1] > 0:
            return (parts[0], parts[1])
        raise ValueError(f"Invalid --cell-size {raw!r}; expected '40' or '40,40'")
    if tokenizer_ref:
        tok_path = Path(tokenizer_ref)
        if tok_path.is_dir():
            try:
                return read_image_cell_size(tok_path)
            except (FileNotFoundError, ValueError):
                pass
    return DEFAULT_INKLING_CELL_SIZE


def count_grid_image_tokens(width: int, height: int, *, cell_w: int = 40, cell_h: int = 40) -> int:
    """Count realized grid image tokens with ceiling padding to cell boundaries."""
    if width <= 0 or height <= 0 or cell_w <= 0 or cell_h <= 0:
        raise ValueError(f"Image and cell dimensions must be positive: {(width, height, cell_w, cell_h)}")
    return math.ceil(width / cell_w) * math.ceil(height / cell_h)


def compute_target_grid_dimensions(
    width: int,
    height: int,
    *,
    max_image_tokens: int,
    cell_w: int = 40,
    cell_h: int = 40,
) -> tuple[int, int, int]:
    """Compute non-upscaling target pixel size (exact cell multiples when >= cell) and token count."""
    if width <= 0 or height <= 0 or cell_w <= 0 or cell_h <= 0 or max_image_tokens <= 0:
        raise ValueError(
            f"All resize parameters must be positive: w={width}, h={height}, "
            f"max_image_tokens={max_image_tokens}, cell=({cell_w}, {cell_h})"
        )

    scale = min(1.0, math.sqrt((max_image_tokens * cell_w * cell_h) / (width * height)))
    cols = max(1, int(math.floor((width * scale) / cell_w)))
    rows = max(1, int(math.floor((height * scale) / cell_h)))
    while cols * rows > max_image_tokens and (cols > 1 or rows > 1):
        if cols >= rows and cols > 1:
            cols -= 1
        elif rows > 1:
            rows -= 1

    target_w = width if width < cell_w else cols * cell_w
    target_h = height if height < cell_h else rows * cell_h
    return target_w, target_h, cols * rows


def resize_image_for_token_budget(
    img: Any,
    *,
    max_image_tokens: int,
    cell_w: int = 40,
    cell_h: int = 40,
) -> tuple[Any, int]:
    """Resize PIL image to exact cell multiples within ``max_image_tokens`` without upscaling."""
    from PIL import Image

    orig_w, orig_h = img.size
    target_w, target_h, tok_count = compute_target_grid_dimensions(
        orig_w, orig_h, max_image_tokens=max_image_tokens, cell_w=cell_w, cell_h=cell_h
    )
    rgb = img if getattr(img, "mode", None) == "RGB" else img.convert("RGB")
    if (orig_w, orig_h) == (target_w, target_h):
        return rgb, tok_count

    resample = _resolve_lanczos_filter(Image)
    try:
        resized = rgb.resize((target_w, target_h), resample=resample)
    finally:
        if rgb is not img and hasattr(rgb, "close"):
            rgb.close()
    return resized, tok_count


def encode_image_for_vlm(
    img: Any,
    *,
    max_image_tokens: int | None = None,
    cell_w: int = 40,
    cell_h: int = 40,
    image_format: str = "PNG",
) -> tuple[bytes, int, int, int]:
    """Optionally budget-resize ``img`` and encode RGB bytes cleanly for VLM transfer/export."""
    fmt = image_format.upper()
    if fmt not in {"PNG", "JPEG"}:
        raise ValueError(f"Unsupported image_format={image_format!r}; expected 'PNG' or 'JPEG'")

    if max_image_tokens is not None:
        prepared, tok_count = resize_image_for_token_budget(
            img, max_image_tokens=max_image_tokens, cell_w=cell_w, cell_h=cell_h
        )
    else:
        prepared = img if getattr(img, "mode", None) == "RGB" else img.convert("RGB")
        w, h = prepared.size
        tok_count = count_grid_image_tokens(w, h, cell_w=cell_w, cell_h=cell_h)

    buf = io.BytesIO()
    try:
        out_w, out_h = prepared.size
        if fmt == "JPEG":
            prepared.save(buf, format="JPEG", quality=95, subsampling=0)
        else:
            prepared.save(buf, format="PNG")
        return buf.getvalue(), int(out_w), int(out_h), int(tok_count)
    finally:
        if prepared is not img and hasattr(prepared, "close"):
            prepared.close()


def measure_max_vision_tokens(
    image_paths: Sequence[Path],
    processor_dir_or_id: str | Path,
    *,
    max_image_tokens: int | None = None,
    cell_size: tuple[int, int] | None = None,
    expected_image_token: str = DEFAULT_IMAGE_TOKEN_STR,
    prompt_text: str = "Transcribe.",
    revision: str | None = None,
) -> dict[str, Any]:
    """Measure realized processor image-token count across sample pages, failing closed on 0 tokens."""
    if not image_paths:
        raise ValueError("measure_max_vision_tokens requires at least one image path")

    from PIL import Image
    from transformers import AutoProcessor

    proc_kw = {"revision": revision} if revision else {}
    proc = AutoProcessor.from_pretrained(str(processor_dir_or_id), **proc_kw)
    tok = getattr(proc, "tokenizer", None)

    img_tid = getattr(proc, "image_token_id", None)
    if img_tid is None and tok is not None:
        img_tid = getattr(tok, "image_token_id", None)
    if img_tid is None and tok is not None and hasattr(tok, "convert_tokens_to_ids"):
        cand_id = tok.convert_tokens_to_ids(expected_image_token)
        unk_id = getattr(tok, "unk_token_id", None)
        if cand_id is not None and int(cand_id) >= 0 and (unk_id is None or int(cand_id) != int(unk_id)):
            img_tid = int(cand_id)
    if img_tid is None or int(img_tid) < 0:
        raise ValueError(
            f"Processor {processor_dir_or_id!r} does not expose a valid image_token_id or {expected_image_token!r}"
        )
    img_tid_int = int(img_tid)

    if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
        mapped = tok.convert_tokens_to_ids(expected_image_token)
        unk_id = getattr(tok, "unk_token_id", None)
        if mapped is not None and int(mapped) >= 0 and (unk_id is None or int(mapped) != int(unk_id)):
            if int(mapped) != img_tid_int:
                raise ValueError(
                    f"Processor image_token_id={img_tid_int} disagrees with {expected_image_token!r}={int(mapped)}"
                )

    if cell_size is not None:
        cell_w, cell_h = cell_size
    elif Path(processor_dir_or_id).is_dir():
        try:
            cell_w, cell_h = read_image_cell_size(Path(processor_dir_or_id))
        except (FileNotFoundError, ValueError):
            cell_w, cell_h = DEFAULT_INKLING_CELL_SIZE
    else:
        ip = getattr(proc, "image_processor", None)
        sz = getattr(ip, "size", None)
        if isinstance(sz, Mapping) and sz.get("width") and sz.get("height"):
            cell_w, cell_h = int(sz["width"]), int(sz["height"])
        else:
            cell_w, cell_h = DEFAULT_INKLING_CELL_SIZE

    counts: list[int] = []
    rendered = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for img_path in image_paths:
        raw_im = Image.open(str(img_path))
        prepared = None
        try:
            if max_image_tokens is not None:
                prepared, _ = resize_image_for_token_budget(
                    raw_im, max_image_tokens=max_image_tokens, cell_w=cell_w, cell_h=cell_h
                )
            else:
                prepared = raw_im if getattr(raw_im, "mode", None) == "RGB" else raw_im.convert("RGB")
            enc = proc(text=[rendered], images=[prepared], return_tensors="pt")
        finally:
            if prepared is not None and prepared is not raw_im and hasattr(prepared, "close"):
                prepared.close()
            if hasattr(raw_im, "close"):
                raw_im.close()

        raw_ids = enc["input_ids"][0]
        ids_list = raw_ids.tolist() if hasattr(raw_ids, "tolist") else list(raw_ids)
        tok_count = sum(1 for tid in ids_list if int(tid) == img_tid_int)
        if tok_count <= 0:
            raise ValueError(
                f"Processor emitted 0 image tokens (image_token_id={img_tid_int}) for sample {img_path}"
            )
        counts.append(tok_count)

    return {
        "image_token_id": img_tid_int,
        "num_samples": len(counts),
        "min_image_tokens": min(counts),
        "max_image_tokens": max(counts),
        "counts": counts,
    }
