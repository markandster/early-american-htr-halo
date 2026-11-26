"""Verification helpers for exported HuggingFace and quantized MLX VLM directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _flatten_nested_floats(v: Any) -> list[float]:
    if hasattr(v, "flatten"):
        return [float(x) for x in v.flatten().tolist()]
    if isinstance(v, (list, tuple)):
        out: list[float] = []
        for item in v:
            out.extend(_flatten_nested_floats(item))
        return out
    return [float(v)]


def verify_exported_vlm_dir(
    model_dir: Path,
    *,
    expected_soft_tokens: int = 1120,
    check_gemma4_resolution: bool = True,
    check_nothink_template: bool = True,
    model_family: str = "gemma4",
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail closed if merged/MLX directory lost resolution, template rendering, or token parity."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Exported model directory not found: {model_dir}")

    checked_fields: dict[str, Any] = {"model_dir": str(model_dir), "model_family": model_family}
    if check_gemma4_resolution and model_family == "gemma4":
        proc_path = model_dir / "processor_config.json"
        if not proc_path.is_file():
            raise FileNotFoundError(f"Missing required {proc_path} in Gemma 4 export directory")
        data = json.loads(proc_path.read_text(encoding="utf-8"))
        if data.get("image_seq_length") != expected_soft_tokens:
            raise ValueError(
                f"{proc_path}: image_seq_length={data.get('image_seq_length')} != {expected_soft_tokens}"
            )
        ip = data.get("image_processor")
        if not isinstance(ip, dict) or ip.get("max_soft_tokens") != expected_soft_tokens:
            raise ValueError(
                f"{proc_path}: image_processor.max_soft_tokens={ip.get('max_soft_tokens') if isinstance(ip, dict) else None} "
                f"!= {expected_soft_tokens}"
            )
        if "image_seq_length" in ip and ip["image_seq_length"] != expected_soft_tokens:
            raise ValueError(
                f"{proc_path}: image_processor.image_seq_length={ip['image_seq_length']} != {expected_soft_tokens}"
            )
        checked_fields["image_seq_length"] = expected_soft_tokens

    if check_nothink_template:
        from jinja2 import BaseLoader, Environment
        from jinja2.ext import Extension

        class _GenTagExt(Extension):
            tags = {"generation"}

            def parse(self, parser: Any) -> Any:
                next(parser.stream)
                return parser.parse_statements(["name:endgeneration"], drop_needle=True)

        tpl_text = ""
        jinja_path = model_dir / "chat_template.jinja"
        tok_cfg_path = model_dir / "tokenizer_config.json"
        if jinja_path.is_file():
            tpl_text = jinja_path.read_text(encoding="utf-8")
        elif tok_cfg_path.is_file():
            tok_data = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            raw = tok_data.get("chat_template")
            tpl_text = raw if isinstance(raw, str) else ""
        if not tpl_text.strip():
            raise ValueError(f"{model_dir}: missing chat_template.jinja and tokenizer_config.json chat_template")
        if "else 0.9" in tpl_text or "Thinking effort level: 0.9" in tpl_text:
            raise ValueError(f"{model_dir}: detected Inkling 0.9 reasoning-effort default instead of 0.0")
        env = Environment(loader=BaseLoader(), extensions=[_GenTagExt])
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        tpl = env.from_string(tpl_text)
        msg1 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}]
        msg_full = msg1 + [{"role": "assistant", "content": [{"type": "text", "text": "Dear Sir"}]}]
        msg2 = msg_full + [{"role": "user", "content": [{"type": "text", "text": "Next page."}]}]
        r1 = tpl.render(messages=msg1, add_generation_prompt=True, enable_thinking=False, bos_token="", eos_token="", tools=None)
        r1_default = tpl.render(messages=msg1, add_generation_prompt=True, bos_token="", eos_token="", tools=None)
        r1_nogen = tpl.render(messages=msg1, add_generation_prompt=False, bos_token="", eos_token="", tools=None)
        r_full = tpl.render(messages=msg_full, add_generation_prompt=False, bos_token="", eos_token="", tools=None)
        r2 = tpl.render(messages=msg2, add_generation_prompt=True, bos_token="", eos_token="", tools=None)
        if not r2.startswith(r1_nogen) or not r_full.startswith(r1) or r1 != r1_default:
            raise ValueError(f"{model_dir}: chat template failed assistant-turn prefix invariance or default-nothink parity")
        if model_family == "gemma4" and ("<|turn>model" not in r1 or "<|channel>thought\n<channel|>" not in r1):
            raise ValueError(f"{model_dir}: rendered Gemma 4 template missing non-thinking <|channel>thought\\n<channel|> sentinel")
        if model_family == "inkling2" and (("Thinking effort level: 0<" not in r1 and "Thinking effort level: 0.0" not in r1) or "0.9" in r1):
            raise ValueError(f"{model_dir}: rendered Inkling 2 template did not default to effort 0")
        checked_fields["chat_template_verified"] = True

    if reference_dir is not None:
        from transformers import AutoTokenizer

        ref_tok = AutoTokenizer.from_pretrained(str(reference_dir))
        exp_tok = AutoTokenizer.from_pretrained(str(model_dir))
        sample = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}]
        sample_full = sample + [{"role": "assistant", "content": [{"type": "text", "text": "Dear Sir"}]}]
        if ref_tok.eos_token_id != exp_tok.eos_token_id:
            raise ValueError(f"{model_dir}: eos_token_id {exp_tok.eos_token_id} != reference {ref_tok.eos_token_id}")
        if (
            ref_tok.apply_chat_template(sample, tokenize=True, add_generation_prompt=True)
            != exp_tok.apply_chat_template(sample, tokenize=True, add_generation_prompt=True)
            or ref_tok.apply_chat_template(sample_full, tokenize=True, add_generation_prompt=False)
            != exp_tok.apply_chat_template(sample_full, tokenize=True, add_generation_prompt=False)
        ):
            raise ValueError(f"{model_dir}: tokenized chat template diverged from {reference_dir}")
        checked_fields["reference_token_parity"] = True

        tf_mod = __import__("transformers")
        auto_proc_cls = getattr(tf_mod, "AutoProcessor", None)
        if auto_proc_cls is not None and (model_dir / "processor_config.json").is_file() and (reference_dir / "processor_config.json").is_file():
            try:
                from PIL import Image
            except ImportError:
                Image = None  # type: ignore[assignment]
            if Image is not None and hasattr(Image, "new"):
                ref_proc = auto_proc_cls.from_pretrained(str(reference_dir))
                exp_proc = auto_proc_cls.from_pretrained(str(model_dir))
                if hasattr(Image, "frombytes"):
                    raw_rgb = bytes(
                        c
                        for y in range(16)
                        for x in range(16)
                        for c in ((x * 16 + y) % 256, (y * 16 + 64) % 256, ((x ^ y) * 16) % 256)
                    )
                    raw_l = bytes((x * 13 + y * 7) % 256 for y in range(16) for x in range(16))
                    probe_imgs = [
                        Image.frombytes("RGB", (16, 16), raw_rgb),
                        Image.frombytes("L", (16, 16), raw_l),
                    ]
                else:
                    probe_imgs = [Image.new("RGB", (16, 16), (255, 255, 255))]
                ref_p = ref_proc.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
                exp_p = exp_proc.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)

                for probe_img in probe_imgs:
                    ref_enc = ref_proc(text=[ref_p], images=[probe_img], return_tensors="pt")
                    exp_enc = exp_proc(text=[exp_p], images=[probe_img], return_tensors="pt")
                    ref_ids = ref_enc["input_ids"][0].tolist() if hasattr(ref_enc["input_ids"][0], "tolist") else list(ref_enc["input_ids"][0])
                    exp_ids = exp_enc["input_ids"][0].tolist() if hasattr(exp_enc["input_ids"][0], "tolist") else list(exp_enc["input_ids"][0])
                    if ref_ids != exp_ids:
                        raise ValueError(f"{model_dir}: multimodal AutoProcessor input_ids diverged from {reference_dir}")
                    if "pixel_values" in ref_enc and "pixel_values" in exp_enc:
                        r_flat, e_flat = _flatten_nested_floats(ref_enc["pixel_values"]), _flatten_nested_floats(exp_enc["pixel_values"])
                        if len(r_flat) != len(e_flat) or any(abs(a - b) > 1e-4 for a, b in zip(r_flat, e_flat)):
                            raise ValueError(f"{model_dir}: image processor pixel_values diverged from {reference_dir}")
                checked_fields["reference_image_parity"] = True

    return checked_fields
