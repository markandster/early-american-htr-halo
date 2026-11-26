"""Unit tests for stdlib vLLM client, engine adapter allow-list validator, and server eval CLI."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import types
from urllib.error import HTTPError

import pytest

from eahtr.vllm_client import (
    VLLMAuthError,
    VLLMClient,
    VLLMResponseError,
    VLLMServerError,
    VLLMTimeoutError,
    read_safetensors_header_keys,
    redact_secret,
    validate_base_url,
    validate_vllm_engine_adapter_dir,
)


def _write_safetensors(path: Path, tensors: dict[str, dict]) -> None:
    raw = json.dumps({"__metadata__": {"format": "pt"}, **tensors}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 16)


def test_engine_adapter_allow_list_accepts_exact_keys_and_rejects_bad_keys(tmp_path: Path) -> None:
    good = tmp_path / "good_adapter"
    good.mkdir()
    (good / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 8}), encoding="utf-8")
    _write_safetensors(
        good / "adapter_model.safetensors",
        {
            "base_model.model.model.llm.layers.0.attn.wo_ud.lora_A.weight": {"dtype": "BF16", "shape": [8, 4608], "data_offsets": [0, 8]},
            "base_model.model.model.llm.layers.0.attn.wo_ud.lora_B.weight": {"dtype": "BF16", "shape": [4608, 8], "data_offsets": [8, 16]},
            "base_model.model.model.llm.layers.0.mlp.experts.w13_weight.lora_A.weight": {"dtype": "BF16", "shape": [384, 8, 4608], "data_offsets": [0, 8]},
            "base_model.model.model.llm.layers.0.mlp.experts.w13_weight.lora_B.weight": {"dtype": "BF16", "shape": [384, 4608, 8], "data_offsets": [8, 16]},
            "base_model.model.model.llm.layers.0.mlp.experts.w2_weight.lora_A.weight": {"dtype": "BF16", "shape": [384, 8, 2304], "data_offsets": [0, 8]},
            "base_model.model.model.llm.layers.0.mlp.experts.w2_weight.lora_B.weight": {"dtype": "BF16", "shape": [384, 4608, 8], "data_offsets": [8, 16]},
        },
    )
    meta = validate_vllm_engine_adapter_dir(good, expected_r=8)
    assert meta["tensor_count"] == 6 and meta["expert_count"] == 384

    # Alpha mismatch fails closed
    (good / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8")
    with pytest.raises(ValueError, match="lora_alpha == r"):
        validate_vllm_engine_adapter_dir(good)
    (good / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 8}), encoding="utf-8")

    # Wrong rank dimension in shape fails closed
    _write_safetensors(
        good / "adapter_model.safetensors",
        {"base_model.model.model.llm.layers.0.attn.wo_ud.lora_A.weight": {"dtype": "BF16", "shape": [16, 4608], "data_offsets": [0, 8]}},
    )
    with pytest.raises(ValueError, match="rank dimension 16 != adapter_config.r"):
        validate_vllm_engine_adapter_dir(good)

    # Known un-exported training keys fail closed
    for bad_key in (
        "base_model.model.model.llm.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
        "base_model.model.model.llm.layers.0.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.llm.layers.0.attn.q_proj.lora_A.weight",
    ):
        _write_safetensors(
            good / "adapter_model.safetensors",
            {bad_key: {"dtype": "BF16", "shape": [8, 64], "data_offsets": [0, 8]}},
        )
        with pytest.raises(ValueError):
            validate_vllm_engine_adapter_dir(good)

    # Corrupt safetensors header fails closed
    bad_st = tmp_path / "bad.safetensors"
    bad_st.write_bytes(struct.pack("<Q", 99999999) + b"{}")
    with pytest.raises(ValueError, match="header length"):
        read_safetensors_header_keys(bad_st)


def test_url_redaction_and_client_retry_and_failure_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-super-secret-token"
    assert "[REDACTED]" in redact_secret(f"Bearer {secret} rejected", secret)
    assert secret not in redact_secret(f"Bearer {secret} rejected", secret)

    assert validate_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    assert validate_base_url("https://vllm.example.org:8443") == "https://vllm.example.org:8443"
    with pytest.raises(ValueError, match="HTTPS"):
        validate_base_url("http://10.20.30.40:8000")
    with pytest.raises(ValueError, match="credentials"):
        validate_base_url("https://user:pass@vllm.example.org")

    monkeypatch.setattr("eahtr.vllm_client.time.sleep", lambda s: None)
    client = VLLMClient("http://127.0.0.1:8000", model="inkling2", api_key=secret, max_retries=2, retry_backoff_s=0.01)

    # 503 then 200 succeeds and sends chat_template_kwargs={"reasoning_effort": 0.0}
    seen_payloads: list[dict] = []

    class _FakeResp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return self._payload

    calls = {"n": 0}

    def _retry_then_ok(req: object, timeout: float = 0) -> _FakeResp:
        calls["n"] += 1
        seen_payloads.append(json.loads(req.data.decode("utf-8")))  # type: ignore[attr-defined]
        if calls["n"] == 1:
            raise HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b"overloaded"))  # type: ignore[attr-defined]
        body = json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "Dear Sir", "reasoning_content": ""}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 2},
        }).encode("utf-8")
        return _FakeResp(body)

    monkeypatch.setattr("eahtr.vllm_client.urlopen", _retry_then_ok)
    out = client.transcribe_page(b"png", prompt_text="Transcribe.")
    assert out["text"] == "Dear Sir" and out["usage"]["prompt_tokens"] == 500
    assert seen_payloads[-1]["chat_template_kwargs"] == {"reasoning_effort": 0.0}
    assert "reasoning_effort" not in seen_payloads[-1]

    # 401 fails immediately without retrying and redacts secret
    def _auth_fail(req: object, timeout: float = 0) -> _FakeResp:
        raise HTTPError(req.full_url, 401, "unauth", {}, io.BytesIO(f"bad key {secret}".encode("utf-8")))  # type: ignore[attr-defined]

    monkeypatch.setattr("eahtr.vllm_client.urlopen", _auth_fail)
    with pytest.raises(VLLMAuthError) as exc_info:
        client.transcribe_page(b"png", prompt_text="Transcribe.")
    assert secret not in str(exc_info.value)

    # Timeout raises VLLMTimeoutError
    def _timeout_fail(req: object, timeout: float = 0) -> _FakeResp:
        raise TimeoutError("read timed out")

    monkeypatch.setattr("eahtr.vllm_client.urlopen", _timeout_fail)
    with pytest.raises(VLLMTimeoutError):
        client.transcribe_page(b"png", prompt_text="Transcribe.")

    # Bad JSON raises VLLMResponseError; persistent 500 raises VLLMServerError
    monkeypatch.setattr("eahtr.vllm_client.urlopen", lambda req, timeout=0: _FakeResp(b"{not-json"))
    with pytest.raises(VLLMResponseError):
        client.transcribe_page(b"png", prompt_text="Transcribe.")
    monkeypatch.setattr(
        "eahtr.vllm_client.urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b"fail"))),  # type: ignore[attr-defined]
    )
    with pytest.raises(VLLMServerError):
        client.transcribe_page(b"png", prompt_text="Transcribe.")


def test_eval_vllm_server_cli_parity_reasoning_leak_and_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_vllm_server", repo_root / "scripts" / "eval_vllm_server.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    img_dir = tmp_path / "pages"
    img_dir.mkdir()
    (img_dir / "p1.png").write_bytes(b"IMG1")
    (img_dir / "p2.png").write_bytes(b"IMG2")
    val_jsonl = tmp_path / "val.jsonl"
    val_jsonl.write_text(
        "\n".join([
            json.dumps({"record_id": "r1", "document_id": "d1", "split": "val", "image_path": "p1.png", "text": "Dear Sir"}),
            json.dumps({"record_id": "r2", "document_id": "d2", "split": "val", "image_path": "p2.png", "text": "Madam"}),
        ]) + "\n",
        encoding="utf-8",
    )

    class _FakeImg:
        def __init__(self, size: tuple[int, int] = (80, 80)) -> None:
            self.size = size
            self.mode = "RGB"

        def convert(self, mode: str) -> "_FakeImg":
            return self

        def resize(self, size: tuple[int, int], resample: object = None) -> "_FakeImg":
            return _FakeImg(size)

        def save(self, buf: object, format: str = "PNG", **kwargs: object) -> None:
            buf.write(b"PNGDATA")  # type: ignore[attr-defined]

        def close(self) -> None:
            pass

    fake_pil = types.ModuleType("PIL")
    fake_pil_img = types.ModuleType("PIL.Image")
    fake_pil_img.open = lambda p: _FakeImg((80, 80))  # type: ignore[attr-defined]
    fake_pil_img.Resampling = types.SimpleNamespace(LANCZOS="LANCZOS")  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_img  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_img)

    class _FakeResp:
        def __init__(self, data: dict) -> None:
            self._data = json.dumps(data).encode("utf-8")

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return self._data

    # 80x80 -> 4 image tokens; returning prompt_tokens=1 < 4 must fail closed
    monkeypatch.setattr(
        "eahtr.vllm_client.urlopen",
        lambda req, timeout=0: _FakeResp({"choices": [{"finish_reason": "stop", "message": {"content": "Dear Sir"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
    )
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_vllm_server.py", "--base-url", "http://127.0.0.1:8000", "--model", "m", "--eval-jsonl", str(val_jsonl), "--image-root", str(img_dir), "--out-json", str(out_json), "--cell-size", "40"],
    )
    with pytest.raises(RuntimeError, match="vision parity failed"):
        cli.main()

    # Valid prompt_tokens passes, detects structured reasoning_content leak, writes median_strict_cer_pct
    replies = iter([
        {"choices": [{"finish_reason": "stop", "message": {"content": "Dear Sir", "reasoning_content": "thought..."}}], "usage": {"prompt_tokens": 24, "completion_tokens": 2}},
        {"choices": [{"finish_reason": "stop", "message": {"content": "Madam", "reasoning_content": ""}}], "usage": {"prompt_tokens": 24, "completion_tokens": 1}},
    ])
    monkeypatch.setattr("eahtr.vllm_client.urlopen", lambda req, timeout=0: _FakeResp(next(replies)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_vllm_server.py", "--base-url", "http://127.0.0.1:8000", "--model", "m", "--eval-jsonl", str(val_jsonl), "--image-root", str(img_dir), "--out-json", str(out_json), "--cell-size", "40", "--swap-images"],
    )
    assert cli.main() == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["scores"]["thinking_leak_pages"] == 1
    assert saved["scores"]["swap_images"] is True
    assert "median_strict_cer_pct" in saved["scores"]

    # Frozen test guard exits 2 without --allow-frozen-test
    test_jsonl = tmp_path / "test_public.jsonl"
    test_jsonl.write_text(val_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_vllm_server.py", "--base-url", "http://127.0.0.1:8000", "--model", "m", "--eval-jsonl", str(test_jsonl), "--out-json", str(out_json)],
    )
    assert cli.main() == 2

    # Missing image_path fails closed inside manifest validation (exit 2)
    bad_img_jsonl = tmp_path / "missing_img.jsonl"
    bad_img_jsonl.write_text(json.dumps({"record_id": "x1", "document_id": "d1", "split": "val", "text": "Dear Sir"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_vllm_server.py", "--base-url", "http://127.0.0.1:8000", "--model", "m", "--eval-jsonl", str(bad_img_jsonl), "--out-json", str(out_json)],
    )
    assert cli.main() == 2

    # With --tokenizer set, prompt_tokens outside tolerance fails closed while aligned count passes
    class _TokParity:
        image_token_id = 200054

        @classmethod
        def from_pretrained(cls, path: str) -> "_TokParity":
            return cls()

        def apply_chat_template(self, msgs: object, tokenize: bool = True, add_generation_prompt: bool = True) -> list[int]:
            return [1, 200054, 2, 3, 4]  # 4 non-image text tokens

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = _TokParity  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    # 80x80 -> 4 img tokens + 4 text tokens = 8 expected; 90 exceeds tol=8
    monkeypatch.setattr(
        "eahtr.vllm_client.urlopen",
        lambda req, timeout=0: _FakeResp({"choices": [{"finish_reason": "stop", "message": {"content": "Dear Sir"}}], "usage": {"prompt_tokens": 90, "completion_tokens": 2}}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_vllm_server.py", "--base-url", "http://127.0.0.1:8000", "--model", "m", "--eval-jsonl", str(val_jsonl), "--image-root", str(img_dir), "--out-json", str(out_json), "--cell-size", "40", "--tokenizer", "stub/ink", "--prompt-token-tolerance", "8"],
    )
    with pytest.raises(RuntimeError, match="prompt_tokens parity drift"):
        cli.main()


def test_halo_launcher_ep_parity_and_fresh_dir_guards(tmp_path: Path) -> None:
    import os
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    non_empty_out = tmp_path / "stale_adapter"
    non_empty_out.mkdir()
    (non_empty_out / "checkpoint-99").mkdir()

    # Non-empty ADAPTER_DIR without RESUME=1 fails fast on both launchers
    for script_name, extra in (("run_gemma4_halo.sh", {"MODEL_SIZE": "26b_a4b", "NUM_GPUS": "8"}), ("run_inkling2_halo.sh", {"NUM_GPUS": "8"})):
        env = {**os.environ, "DATASET_DIR": str(ds_dir), "ADAPTER_DIR": str(non_empty_out), **extra}
        res = subprocess.run(["bash", str(repo_root / "scripts" / script_name)], env=env, capture_output=True, text=True)
        assert res.returncode != 0 and "non-empty" in res.stderr

    # Existing ENGINE_ADAPTER_DIR fails fast on Inkling launcher
    existing_engine = tmp_path / "existing_engine"
    existing_engine.mkdir()
    fresh_adapter = tmp_path / "fresh_inkling_adapter"
    env_eng = {**os.environ, "DATASET_DIR": str(ds_dir), "ADAPTER_DIR": str(fresh_adapter), "ENGINE_ADAPTER_DIR": str(existing_engine), "NUM_GPUS": "8"}
    res_eng = subprocess.run(["bash", str(repo_root / "scripts" / "run_inkling2_halo.sh")], env=env_eng, capture_output=True, text=True)
    assert res_eng.returncode != 0 and "already exists" in res_eng.stderr

    # EP=4 on NUM_GPUS=8 fails closed on both launchers before touching Halo or weights
    for script_name, extra in (("run_gemma4_halo.sh", {"MODEL_SIZE": "26b_a4b"}), ("run_inkling2_halo.sh", {})):
        fresh_dir = tmp_path / f"ep_{script_name}"
        fresh_eng = tmp_path / f"eng_{script_name}"
        res_ep = subprocess.run(
            ["bash", str(repo_root / "scripts" / script_name)],
            env={
                **os.environ,
                "PYTHON_BIN": sys.executable,
                "WORK_DIR": str(tmp_path / f"work_{script_name}"),
                "DATASET_DIR": str(ds_dir),
                "ADAPTER_DIR": str(fresh_dir),
                "ENGINE_ADAPTER_DIR": str(fresh_eng),
                "EP_SIZE": "4",
                "NUM_GPUS": "8",
                **extra,
            },
            capture_output=True,
            text=True,
        )
        assert res_ep.returncode != 0 and "Invalid GPU count" in res_ep.stderr
