"""Stdlib-only vLLM OpenAI-compatible VLM client and fail-closed engine adapter validator."""

from __future__ import annotations

import base64
import ipaddress
import json
from pathlib import Path
import re
import socket
import struct
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

_INKLING_WO_RE = re.compile(r"^base_model\.model\.model\.llm\.layers\.(\d+)\.attn\.wo_ud\.lora_([AB])\.weight$")
_INKLING_W13_RE = re.compile(r"^base_model\.model\.model\.llm\.layers\.(\d+)\.mlp\.experts\.w13_weight\.lora_([AB])\.weight$")
_INKLING_W2_RE = re.compile(r"^base_model\.model\.model\.llm\.layers\.(\d+)\.mlp\.experts\.w2_weight\.lora_([AB])\.weight$")
_UNEXPORTED_HINTS = (".experts.gate_up_proj.", ".self_attn.", ".o_proj.", ".down_proj.")


class VLLMClientError(RuntimeError):
    """Base error raised by the stdlib vLLM client."""


class VLLMAuthError(VLLMClientError):
    """Raised on HTTP 401/403 authentication or authorization failure."""


class VLLMTimeoutError(VLLMClientError):
    """Raised when vLLM requests exceed timeout bounds after retries."""


class VLLMServerError(VLLMClientError):
    """Raised on server HTTP 4xx/5xx or transport failure."""


class VLLMResponseError(VLLMClientError):
    """Raised when the vLLM server returns malformed JSON or missing choice fields."""


def redact_secret(text: str, secret: str | None) -> str:
    """Replace any occurrence of ``secret`` inside ``text`` with ``[REDACTED]``."""
    raw = str(text)
    if secret and len(secret) > 0:
        return raw.replace(secret, "[REDACTED]")
    return raw


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_base_url(base_url: str, *, allow_insecure_remote: bool = False) -> str:
    """Validate an OpenAI-compatible vLLM base URL and enforce HTTPS for non-loopback hosts."""
    cleaned = str(base_url or "").strip()
    if not cleaned:
        raise ValueError("vLLM base_url must not be empty")
    parsed = urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme {scheme!r} for vLLM base_url; expected 'http' or 'https'")
    if parsed.username or parsed.password:
        raise ValueError("vLLM base_url must not embed credentials")
    host = parsed.hostname
    if not host:
        raise ValueError(f"vLLM base_url missing hostname: {cleaned!r}")
    if scheme == "http" and not _is_loopback_host(host) and not allow_insecure_remote:
        raise ValueError(
            f"Refusing insecure remote HTTP vLLM base_url {cleaned!r}; use HTTPS or pass allow_insecure_remote=True"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(f"vLLM base_url must not include query parameters or fragments: {cleaned!r}")
    return f"{scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def read_safetensors_header_keys(path: Path) -> dict[str, dict[str, Any]]:
    """Read a safetensors file's JSON tensor header without loading tensor payloads."""
    if not path.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {path}")
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"Invalid safetensors file (too small: {file_size} bytes): {path}")
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        if header_len == 0 or header_len > _MAX_SAFETENSORS_HEADER_BYTES or (8 + header_len) > file_size:
            raise ValueError(f"Invalid safetensors header length {header_len} in {path}")
        raw_header = fh.read(header_len)
    try:
        parsed = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupt safetensors UTF-8 JSON header in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Safetensors header must be a JSON object in {path}")
    return {k: v for k, v in parsed.items() if k != "__metadata__" and isinstance(v, dict)}


def _validate_tensor_rank_shape(
    key: str,
    side: str,
    shape: Any,
    r: int,
    *,
    is_expert: bool,
    expert_counts: set[int],
) -> None:
    if not isinstance(shape, list) or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ValueError(f"Invalid tensor shape {shape!r} for {key}")
    if is_expert:
        if len(shape) != 3:
            raise ValueError(f"Expected 3D expert LoRA tensor [E, *, *] for {key}, got {shape!r}")
        expert_counts.add(int(shape[0]))
        expected_r_dim = int(shape[1]) if side == "A" else int(shape[2])
    else:
        if len(shape) != 2:
            raise ValueError(f"Expected 2D attention LoRA tensor for {key}, got {shape!r}")
        expected_r_dim = int(shape[0]) if side == "A" else int(shape[1])
    if expected_r_dim != r:
        raise ValueError(f"Adapter tensor {key} rank dimension {expected_r_dim} != adapter_config.r ({r})")


def validate_vllm_engine_adapter_dir(
    adapter_dir: Path,
    *,
    expected_family: str = "inkling2",
    expected_r: int | None = None,
) -> dict[str, Any]:
    """Verify exported vLLM engine adapter config and strict allow-listed tensor keys/shapes."""
    if expected_family != "inkling2":
        raise ValueError(f"Unsupported engine adapter family {expected_family!r}; expected 'inkling2'")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Engine adapter directory not found: {adapter_dir}")

    cfg_path = adapter_dir / "adapter_config.json"
    st_path = adapter_dir / "adapter_model.safetensors"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing required {cfg_path}")
    if not st_path.is_file():
        raise FileNotFoundError(f"Missing required {st_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    r = cfg.get("r")
    lora_alpha = cfg.get("lora_alpha")
    if not isinstance(r, int) or isinstance(r, bool) or r <= 0:
        raise ValueError(f"{cfg_path}: invalid LoRA rank r={r!r}")
    if expected_r is not None and r != expected_r:
        raise ValueError(f"{cfg_path}: LoRA rank r={r} != expected {expected_r}")
    if lora_alpha != r:
        raise ValueError(
            f"{cfg_path}: exported vLLM engine adapter must have lora_alpha == r ({lora_alpha!r} != {r})"
        )

    tensors = read_safetensors_header_keys(st_path)
    if not tensors:
        raise ValueError(f"{st_path}: no LoRA weight tensors present")

    layers_seen: set[int] = set()
    expert_counts: set[int] = set()
    for key, meta in tensors.items():
        if any(bad in key for bad in _UNEXPORTED_HINTS):
            raise ValueError(
                f"{st_path}: detected un-exported training-format key {key!r}; run Halo's export_engine_adapter.py first"
            )
        shape = meta.get("shape")
        if m := _INKLING_WO_RE.match(key):
            layers_seen.add(int(m.group(1)))
            _validate_tensor_rank_shape(key, m.group(2), shape, r, is_expert=False, expert_counts=expert_counts)
        elif m := _INKLING_W13_RE.match(key):
            layers_seen.add(int(m.group(1)))
            _validate_tensor_rank_shape(key, m.group(2), shape, r, is_expert=True, expert_counts=expert_counts)
        elif m := _INKLING_W2_RE.match(key):
            layers_seen.add(int(m.group(1)))
            _validate_tensor_rank_shape(key, m.group(2), shape, r, is_expert=True, expert_counts=expert_counts)
        else:
            raise ValueError(f"{st_path}: unexpected tensor key outside Inkling engine allow-list: {key!r}")

    if len(expert_counts) > 1:
        raise ValueError(f"{st_path}: inconsistent expert counts across tensors: {sorted(expert_counts)}")

    return {
        "adapter_dir": str(adapter_dir),
        "r": r,
        "lora_alpha": lora_alpha,
        "tensor_count": len(tensors),
        "layer_count": len(layers_seen),
        "expert_count": next(iter(expert_counts), 0),
    }


def _parse_chat_completion_response(raw_body: bytes, secret: str | None) -> dict[str, Any]:
    try:
        data = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VLLMResponseError(redact_secret(f"Invalid JSON from vLLM server: {exc}", secret)) from exc
    if not isinstance(data, dict):
        raise VLLMResponseError("vLLM response payload must be a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VLLMResponseError("vLLM response missing non-empty choices[0]")
    choice = choices[0]
    msg = choice.get("message")
    if not isinstance(msg, dict) or "content" not in msg:
        raise VLLMResponseError("vLLM response choice[0] missing message.content")

    content_val = msg.get("content")
    if isinstance(content_val, list):
        text_out = "".join(str(part.get("text", "")) for part in content_val if isinstance(part, dict))
    elif content_val is None:
        text_out = ""
    else:
        text_out = str(content_val)

    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "text": text_out,
        "reasoning_content": None if reasoning is None else str(reasoning),
        "finish_reason": str(choice.get("finish_reason") or "stop"),
        "usage": dict(usage),
    }


class VLLMClient:
    """Bounded-retry OpenAI chat/completions client for multimodal HTR serving."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        allow_insecure_remote: bool = False,
    ) -> None:
        if not str(model or "").strip():
            raise ValueError("model name must not be empty")
        if timeout_s <= 0 or max_retries < 1:
            raise ValueError(f"Invalid timeout_s={timeout_s} or max_retries={max_retries}")
        self.base_url = validate_base_url(base_url, allow_insecure_remote=allow_insecure_remote)
        self.model = str(model).strip()
        self._api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)

    def transcribe_page(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        prompt_text: str,
        max_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        reasoning_effort: float = 0.0,
    ) -> dict[str, Any]:
        """Send one encoded manuscript image and return text, reasoning, finish_reason, and usage."""
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"reasoning_effort": float(reasoning_effort)},
        }
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = float(repetition_penalty)

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        endpoint = f"{self.base_url}/v1/chat/completions"

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                return _parse_chat_completion_response(raw, self._api_key)
            except HTTPError as exc:
                err_detail = ""
                try:
                    err_detail = exc.read().decode("utf-8", errors="replace")[:400]
                except OSError:
                    err_detail = str(exc)
                msg = redact_secret(f"vLLM HTTP {exc.code} at {endpoint}: {err_detail}", self._api_key)
                if exc.code in {401, 403}:
                    raise VLLMAuthError(msg) from exc
                last_err = VLLMServerError(msg)
                if exc.code not in _TRANSIENT_HTTP_STATUS or attempt >= self.max_retries:
                    raise last_err from exc
            except (TimeoutError, socket.timeout) as exc:
                msg = redact_secret(f"vLLM request timed out after {self.timeout_s}s at {endpoint}", self._api_key)
                last_err = VLLMTimeoutError(msg)
                if attempt >= self.max_retries:
                    raise last_err from exc
            except URLError as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    msg = redact_secret(f"vLLM request timed out after {self.timeout_s}s at {endpoint}", self._api_key)
                    last_err = VLLMTimeoutError(msg)
                else:
                    msg = redact_secret(f"vLLM transport error at {endpoint}: {reason}", self._api_key)
                    last_err = VLLMServerError(msg)
                if attempt >= self.max_retries:
                    raise last_err from exc

            if attempt < self.max_retries and self.retry_backoff_s > 0:
                time.sleep(self.retry_backoff_s * attempt)

        assert last_err is not None
        raise last_err
