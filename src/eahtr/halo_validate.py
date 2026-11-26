"""Two-tier configuration validator for White Circle Halo recipes.

Tier 1 validates YAML 1.2 boolean safety, VLM/EP constraints, and SMPO/DPO/SFT keys
without external dependencies. Tier 2 imports an external whitecircle/halo checkout via
``--halo-repo`` (preserving LicenseRef-Halo clean-room separation) inside Python 3.12.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import re
import sys
import types
from typing import Any

import yaml

REFUSED_STARTUP_KEYS: frozenset[str] = frozenset(
    {
        "assistant_only_loss",
        "completion_only_loss",
        "max_pixels",
        "warmup_ratio",
        "num_rollout_workers",
    }
)
ALLOWED_RLVR_REWARD_SOURCES: frozenset[str] = frozenset(
    {"accuracy", "format", "judge", "reward_model", "whitecircle", "environment"}
)
_YAML11_BOOL_RE = re.compile(r"^\s*[^#\-\s][^:]*:\s*(yes|no|on|off)\s*(?:#.*)?$", re.IGNORECASE)


def _find_repo_root(config_dir: Path, repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root
    resolved = config_dir.resolve()
    return resolved.parents[1] if resolved.parent.name == "configs" else resolved


def _check_yaml11_booleans(yaml_path: Path) -> None:
    for lineno, line in enumerate(yaml_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _YAML11_BOOL_RE.match(line):
            raise ValueError(
                f"{yaml_path.name}:{lineno}: unquoted YAML 1.1 boolean value is forbidden "
                "(use true/false or quote string values explicitly)"
            )


def _list_config_files(config_dir: Path) -> list[Path]:
    files = sorted(
        p for p in config_dir.glob("*.yaml") if not p.name.startswith("._") and p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No Halo YAML configs found under {config_dir}")
    return files


def _validate_single_tier1(yaml_path: Path, repo_root: Path) -> dict[str, Any]:
    _check_yaml11_booleans(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"{yaml_path.name}: expected non-empty YAML mapping")

    bad_keys = sorted(REFUSED_STARTUP_KEYS & set(cfg))
    if bad_keys:
        raise ValueError(f"{yaml_path.name}: contains Halo startup-refused key(s): {bad_keys}")

    if cfg.get("bf16") is not True:
        raise ValueError(f"{yaml_path.name}: must explicitly set bf16: true")
    if "test_size" in cfg and cfg["test_size"] is not None:
        raise ValueError(f"{yaml_path.name}: must omit test_size to preserve writer-disjoint splits")

    if "chat_template" in cfg and cfg["chat_template"] is not None:
        tpl_str = str(cfg["chat_template"])
        if tpl_str.endswith((".jinja", ".jinja2", ".j2")):
            tpl_path = Path(tpl_str)
            if not tpl_path.is_absolute():
                tpl_path = repo_root / tpl_path
            if not tpl_path.is_file():
                raise FileNotFoundError(f"{yaml_path.name}: missing chat_template file {tpl_path}")

    ep_size = int(cfg.get("expert_parallel_size", 1) or 1)
    if ep_size > 1 and cfg.get("use_peft"):
        if cfg.get("lora_target_modules") == "all-linear":
            raise ValueError(f"{yaml_path.name}: all-linear is forbidden when expert_parallel_size > 1")
        if int(cfg.get("lora_r", 0) or 0) % 8 != 0:
            raise ValueError(f"{yaml_path.name}: lora_r must be divisible by 8 under EP > 1")

    name = yaml_path.name
    if "inkling2" in name and cfg.get("pad_token") != "<|endoftext|>":
        raise ValueError(f"{name}: Inkling 2 configs must set pad_token: '<|endoftext|>'")
    if name.startswith("inkling2_small_"):
        if ep_size < 2:
            raise ValueError(f"{name}: Inkling-2-Small requires expert_parallel_size >= 2")
        if cfg.get("frozen_quantized_experts") is not True:
            raise ValueError(f"{name}: Inkling-2-Small requires frozen_quantized_experts: true")
        targets = cfg.get("lora_target_modules") or []
        if "o_proj" not in targets:
            raise ValueError(f"{name}: Inkling-2-Small requires o_proj alongside expert LoRA targets")

    if "_smpo" in name or cfg.get("loss_type") == "smooth_lower_bound":
        if cfg.get("loss_type") != "smooth_lower_bound":
            raise ValueError(f"{name}: expected loss_type: smooth_lower_bound")
        if cfg.get("use_margin_schedule") is not True:
            raise ValueError(f"{name}: expected use_margin_schedule: true")
        for forbidden in ("images_field", "truncation_mode"):
            if forbidden in cfg:
                raise ValueError(f"{name}: must omit {forbidden} in Halo VLM SMPO")
        if cfg.get("use_peft") and cfg.get("freeze_vision_tower") is True:
            raise ValueError(f"{name}: cannot combine use_peft: true with freeze_vision_tower: true")
        if cfg.get("use_peft") and float(cfg.get("lora_dropout", -1.0)) != 0.0:
            raise ValueError(f"{name}: SMPO PEFT requires lora_dropout: 0.0")
        if cfg.get("use_peft") and float(cfg.get("learning_rate", 1.0)) > 5.0e-6:
            raise ValueError(f"{name}: SMPO PEFT requires learning_rate <= 5.0e-6")
        if float(cfg.get("chosen_sft_ratio", 0.0)) != 1.0:
            raise ValueError(f"{name}: SMPO requires chosen_sft_ratio: 1.0")

    if "_dpo" in name:
        if cfg.get("images_field") != "images":
            raise ValueError(f"{name}: VLM DPO must set images_field: images")
        if cfg.get("precompute_ref_log_probs") is True:
            raise ValueError(f"{name}: VLM DPO refuses precompute_ref_log_probs: true")
        for forbidden_dpo_key in ("generation_max_prompt_length", "max_prompt_length"):
            if forbidden_dpo_key in cfg:
                raise ValueError(f"{name}: Halo DPO refuses {forbidden_dpo_key}")

    if name == "archival_grpo_online.yaml":
        if "gradient_checkpointing_kwargs" in cfg and cfg.get("gradient_checkpointing") is not True:
            raise ValueError(f"{name}: gradient_checkpointing_kwargs requires gradient_checkpointing: true")
        for term in cfg.get("rewards") or []:
            src = term.get("source") if isinstance(term, dict) else None
            if src not in ALLOWED_RLVR_REWARD_SOURCES:
                raise ValueError(f"{name}: invalid Halo GRPO reward source {src!r}")

    return cfg


def validate_yaml_configs_tier1(config_dir: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Run static schema and policy checks on every YAML file under ``config_dir``."""
    resolved_root = _find_repo_root(config_dir, repo_root)
    files = _list_config_files(config_dir)
    validated: list[str] = []
    for yaml_path in files:
        _validate_single_tier1(yaml_path, resolved_root)
        validated.append(yaml_path.name)
    return {"tier": 1, "validated_files": validated, "count": len(validated)}


def _trainer_spec_for_config(name: str) -> str:
    if name.endswith("_sft.yaml"):
        return "sft"
    if "_vlm_smpo" in name or name.endswith("_smpo.yaml"):
        return "smpo"
    if "_vlm_dpo" in name or name.endswith("_dpo.yaml"):
        return "dpo"
    if "offline_grpo" in name or "_grpo_offline" in name:
        return "grpo_offline"
    if "_grpo_online" in name:
        return "grpo_online"
    raise ValueError(f"Unrecognized Halo trainer family for config {name!r}")


def _install_cpu_gpu_stubs() -> dict[str, Any]:
    """Register minimal stubs for absent CUDA-only packages during CPU config parsing."""
    class _DummySymbol:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> "_DummySymbol":
            return self

    class _FlexMod(types.ModuleType):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.__path__ = []

        def __getattr__(self, name: str) -> Any:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _DummySymbol

    modules: dict[str, Any] = {}
    if importlib.util.find_spec("triton") is None:
        noop = lambda *a, **kw: a[0] if len(a) == 1 and callable(a[0]) and not kw else (lambda fn: fn)
        triton = _FlexMod("triton")
        for attr in ("jit", "autotune", "heuristics", "constexpr_function"):
            setattr(triton, attr, noop)
        triton.cdiv = lambda a, b: (a + b - 1) // b
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length() if n > 0 else 1
        t_subs = (
            "triton.language", "triton.language.extra", "triton.language.extra.libdevice",
            "triton.runtime", "triton.runtime.jit", "triton.runtime.autotuner",
            "triton.errors", "triton.compiler", "triton.backends",
        )
        modules.update({"triton": triton, **{s: _FlexMod(s) for s in t_subs}})
        triton.language, triton.runtime, triton.errors = (
            modules["triton.language"], modules["triton.runtime"], modules["triton.errors"],
        )
        modules["triton.language.extra.libdevice"].tanh = lambda x: x
        modules["triton.runtime.jit"].JITFunction = type("JITFunction", (), {})
        modules["triton.runtime.autotuner"].Autotuner = type("Autotuner", (), {})
        modules["triton.errors"].TritonError = type("TritonError", (Exception,), {})
    if importlib.util.find_spec("fla") is None:
        modules.update({"fla": _FlexMod("fla"), "fla.modules": _FlexMod("fla.modules")})
    if importlib.util.find_spec("liger_kernel") is None:
        l_subs = (
            "liger_kernel", "liger_kernel.transformers", "liger_kernel.transformers.functional",
            "liger_kernel.transformers.model", "liger_kernel.transformers.model.loss_utils",
            "liger_kernel.transformers.model.output_classes", "liger_kernel.transformers.rope",
            "liger_kernel.transformers.auto_model", "liger_kernel.ops",
        )
        modules.update({s: _FlexMod(s) for s in l_subs})
        _noop_applier = lambda *a, **kw: None
        modules["liger_kernel.transformers.auto_model"].MODEL_TYPE_TO_APPLY_LIGER_FN = type(
            "_LigerMap", (dict,), {"get": lambda self, k, d=None: _noop_applier}
        )()

    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    return saved


def _restore_modules(saved: dict[str, Any]) -> None:
    for key, orig in saved.items():
        if orig is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = orig


def _make_cpu_trainer_stub(
    cls_name: str,
    bounded_row_length_fn: Any,
    *,
    supports_cp: bool,
    pp_refusal: str | None,
    supports_lowp: bool,
    forward_rows: int,
    generation_shaped: bool,
) -> type:
    class _CPUTraitDescriptor:
        SUPPORTS_CP = supports_cp
        PP_REFUSAL = pp_refusal
        SUPPORTS_LOWP_PRECISION = supports_lowp
        FORWARD_ROWS_PER_EXAMPLE = forward_rows
        GENERATION_SHAPED_ROWS = generation_shaped

        @classmethod
        def declared_row_length(cls, training_config: Any, *sources: Any) -> int | None:
            return bounded_row_length_fn(training_config, *sources)

        @classmethod
        def reject_undeclared_axes(cls, *, cp_size: int, pp_size: int) -> None:
            if cp_size > 1 and not cls.SUPPORTS_CP:
                raise ValueError(
                    f"{cls.__name__} does not support Context Parallelism (CP). Use Expert Parallelism (EP) "
                    "and/or Tensor Parallelism (TP) instead, or remove context_parallel_size."
                )
            if pp_size > 1 and cls.PP_REFUSAL is not None:
                raise ValueError(
                    f"{cls.__name__} does not support Pipeline Parallelism (PP): {cls.PP_REFUSAL}. "
                    "Use Expert Parallelism (EP) and/or Tensor Parallelism (TP) instead, or remove "
                    "pipeline_parallel_size."
                )

    _CPUTraitDescriptor.__name__ = cls_name
    _CPUTraitDescriptor.__qualname__ = cls_name
    return _CPUTraitDescriptor


def _resolve_halo_imports() -> dict[str, Any]:
    try:
        trl = importlib.import_module("trl")
        parser_mod = importlib.import_module("halo.configs.parser")
        par_mod = importlib.import_module("halo.configs.parallelism_arguments")
        script_args_mod = importlib.import_module("halo.configs.script_arguments")
        sft_cfg_mod = importlib.import_module("halo.configs.sft")
        pref_cfg_mod = importlib.import_module("halo.configs.preference")
        cfg_val_mod = importlib.import_module("halo.config_validation")
        sft_rules_mod = importlib.import_module("halo.trainers.sft_rules")
        ctx_win_mod = importlib.import_module("halo.models.introspection.context_window")

        saved_gpu = _install_cpu_gpu_stubs()
        try:
            data_setup_mod = importlib.import_module("halo.entrypoints.data_setup")
            grpo_off_mod = importlib.import_module("halo.configs.grpo_offline")
            grpo_on_mod = importlib.import_module("halo.configs.grpo_online")
        finally:
            if saved_gpu:
                _restore_modules(saved_gpu)
    except Exception as exc:
        raise RuntimeError(
            "Tier-2 Halo validation failed importing Halo modules/dependencies; ensure Python 3.12 "
            "has torch, transformers, trl, peft, ruamel.yaml, datasets, accelerate, pydantic, "
            f"huggingface-hub, pyarrow, pillow, and wandb installed: {type(exc).__name__}: {exc}"
        ) from exc

    _pp_online = (
        "a pipeline stage holds only a contiguous subset of the layers, so this trainer's "
        "loss cannot be expressed as a last-stage function of logits and labels"
    )
    cpu_traits: dict[str, tuple[str, str, tuple[bool, str | None, bool, int, bool]]] = {
        "DistributedSFTTrainer": ("halo.trainers.sft", "DistributedSFTTrainer", (True, None, True, 1, False)),
        "SMPOTrainer": ("halo.trainers.preference.smpo", "SMPOTrainer", (True, None, False, 2, False)),
        "DistributedDPOTrainer": ("halo.trainers.preference.dpo", "DistributedDPOTrainer", (False, None, False, 2, False)),
        "OfflineGRPOTrainer": ("halo.trainers.grpo.offline.trainer", "OfflineGRPOTrainer", (True, None, False, 1, False)),
        "DistributedOnlineGRPOTrainer": ("halo.trainers.grpo.online", "DistributedOnlineGRPOTrainer", (False, _pp_online, False, 1, True)),
    }

    resolved_trainers: dict[str, type] = {}
    used_cpu_fallback = False
    for key, (mod_path, cls_name, (s_cp, pp_ref, s_lowp, f_rows, g_rows)) in cpu_traits.items():
        try:
            resolved_trainers[key] = getattr(importlib.import_module(mod_path), cls_name)
        except (ImportError, TypeError):
            used_cpu_fallback = True
            resolved_trainers[key] = _make_cpu_trainer_stub(
                cls_name,
                ctx_win_mod.bounded_row_length,
                supports_cp=s_cp,
                pp_refusal=pp_ref,
                supports_lowp=s_lowp,
                forward_rows=f_rows,
                generation_shaped=g_rows,
            )

    return {
        "HaloArgumentParser": parser_mod.HaloArgumentParser,
        "ModelArguments": script_args_mod.ModelArguments,
        "ParallelismArguments": par_mod.ParallelismArguments,
        "parallelism_config_from_args": par_mod.parallelism_config_from_args,
        "reject_frozen_experts_without_peft_adapters": par_mod.reject_frozen_experts_without_peft_adapters,
        "reject_non_default_args": cfg_val_mod.reject_non_default_args,
        "reject_trl_dataset_prep_args": data_setup_mod.reject_trl_dataset_prep_args,
        "reject_unsupported_sft_data_options": sft_rules_mod.reject_unsupported_sft_data_options,
        "SFTScriptArguments": sft_cfg_mod.SFTScriptArguments,
        "SFTConfig": trl.SFTConfig,
        "SMPOScriptArguments": pref_cfg_mod.SMPOScriptArguments,
        "SMPOConfig": pref_cfg_mod.SMPOConfig,
        "DPOScriptArguments": pref_cfg_mod.DPOScriptArguments,
        "DPOConfig": trl.DPOConfig,
        "OfflineGRPOScriptArguments": grpo_off_mod.OfflineGRPOScriptArguments,
        "OfflineGRPOConfig": grpo_off_mod.OfflineGRPOConfig,
        "OnlineGRPOScriptArguments": grpo_on_mod.OnlineGRPOScriptArguments,
        "GRPOConfig": trl.GRPOConfig,
        "used_cpu_trainer_fallback": used_cpu_fallback,
        **resolved_trainers,
    }


def validate_yaml_configs_tier2(
    config_dir: Path,
    halo_repo: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate every config through an external whitecircle/halo checkout's real parser."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Tier-2 Halo validation requires Python 3.12 (whitecircle/halo pin), "
            f"running {sys.version_info[0]}.{sys.version_info[1]}"
        )
    resolved_halo = halo_repo.resolve()
    if not resolved_halo.is_dir() or not (resolved_halo / "halo").is_dir():
        raise FileNotFoundError(
            f"Invalid --halo-repo path {resolved_halo}: expected an external whitecircle/halo checkout"
        )

    tier1_res = validate_yaml_configs_tier1(config_dir, repo_root=repo_root)
    resolved_root = _find_repo_root(config_dir, repo_root)

    added_sys_path = False
    if str(resolved_halo) not in sys.path:
        sys.path.insert(0, str(resolved_halo))
        added_sys_path = True

    try:
        h = _resolve_halo_imports()
        ta_mod = importlib.import_module("transformers.training_args")
        pc_mod = importlib.import_module("halo.distributed.parallelism_config")
        orig_bf16_check = ta_mod.is_torch_bf16_gpu_available
        orig_gws, orig_lws = pc_mod.get_global_world_size, pc_mod.get_local_world_size
        if not orig_bf16_check():
            ta_mod.is_torch_bf16_gpu_available = lambda: True
        families = {
            "sft": ((h["SFTScriptArguments"], h["SFTConfig"], h["ModelArguments"], h["ParallelismArguments"]), h["DistributedSFTTrainer"]),
            "smpo": ((h["SMPOScriptArguments"], h["SMPOConfig"], h["ModelArguments"], h["ParallelismArguments"]), h["SMPOTrainer"]),
            "dpo": ((h["DPOScriptArguments"], h["DPOConfig"], h["ModelArguments"], h["ParallelismArguments"]), h["DistributedDPOTrainer"]),
            "grpo_offline": ((h["OfflineGRPOScriptArguments"], h["OfflineGRPOConfig"], h["ModelArguments"], h["ParallelismArguments"]), h["OfflineGRPOTrainer"]),
            "grpo_online": ((h["OnlineGRPOScriptArguments"], h["GRPOConfig"], h["ModelArguments"], h["ParallelismArguments"]), h["DistributedOnlineGRPOTrainer"]),
        }

        validated: list[str] = []
        try:
            for yaml_path in _list_config_files(config_dir):
                kind = _trainer_spec_for_config(yaml_path.name)
                dc_tuple, trainer_cls = families[kind]
                parser = h["HaloArgumentParser"](dc_tuple)
                values = parser.load_values([str(yaml_path)])
                if isinstance(values.get("chat_template"), str) and values["chat_template"].endswith(
                    (".jinja", ".jinja2", ".j2")
                ):
                    tpl = Path(values["chat_template"])
                    if not tpl.is_absolute():
                        values["chat_template"] = str((resolved_root / tpl).resolve())

                script_args, train_cfg, model_cfg, dist_args = parser.parse_dict(values)
                h["reject_frozen_experts_without_peft_adapters"](dist_args, model_cfg)
                if kind == "sft":
                    h["reject_unsupported_sft_data_options"](train_cfg, dist_args)
                    h["reject_trl_dataset_prep_args"]("Halo SFT", train_cfg)
                elif kind == "smpo" and "_vlm_smpo" in yaml_path.name:
                    h["reject_non_default_args"]("SMPO VLM mode", train_cfg, "truncation_mode")

                target_world = max(
                    1,
                    dist_args.expert_parallel_size
                    * dist_args.expert_tensor_parallel_size
                    * dist_args.pipeline_parallel_size
                    * max(dist_args.tensor_parallel_size, dist_args.context_parallel_size),
                )
                pc_mod.get_global_world_size = lambda w=target_world: w
                pc_mod.get_local_world_size = lambda w=target_world: w
                h["parallelism_config_from_args"](
                    dist_args,
                    trainer_cls=trainer_cls,
                    training_config=train_cfg,
                    supports_init_from_scratch=(kind == "sft"),
                    row_sources=(script_args,) if kind == "grpo_online" else (),
                )
                validated.append(yaml_path.name)
        finally:
            ta_mod.is_torch_bf16_gpu_available = orig_bf16_check
            pc_mod.get_global_world_size, pc_mod.get_local_world_size = orig_gws, orig_lws

        return {
            "tier": 2,
            "halo_repo": str(resolved_halo),
            "validated_files": validated,
            "count": len(validated),
            "used_cpu_trainer_fallback": bool(h.get("used_cpu_trainer_fallback", False)),
            "tier1": tier1_res,
        }
    finally:
        if added_sys_path and str(resolved_halo) in sys.path:
            sys.path.remove(str(resolved_halo))
