"""CPU-only stub helpers for offline Halo config parsing without CUDA packages."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any


def install_cpu_gpu_stubs() -> dict[str, Any]:
    """Register minimal in-memory stubs for optional CUDA packages during CPU validation."""
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
            "triton.language",
            "triton.language.extra",
            "triton.language.extra.libdevice",
            "triton.runtime",
            "triton.runtime.jit",
            "triton.runtime.autotuner",
            "triton.errors",
            "triton.compiler",
            "triton.backends",
        )
        modules.update({"triton": triton, **{s: _FlexMod(s) for s in t_subs}})
        triton.language, triton.runtime, triton.errors = (
            modules["triton.language"],
            modules["triton.runtime"],
            modules["triton.errors"],
        )
        modules["triton.language.extra.libdevice"].tanh = lambda x: x
        modules["triton.runtime.jit"].JITFunction = type("JITFunction", (), {})
        modules["triton.runtime.autotuner"].Autotuner = type("Autotuner", (), {})
        modules["triton.errors"].TritonError = type("TritonError", (Exception,), {})
    if importlib.util.find_spec("fla") is None:
        modules.update({"fla": _FlexMod("fla"), "fla.modules": _FlexMod("fla.modules")})
    if importlib.util.find_spec("liger_kernel") is None:
        l_subs = (
            "liger_kernel",
            "liger_kernel.transformers",
            "liger_kernel.transformers.functional",
            "liger_kernel.transformers.model",
            "liger_kernel.transformers.model.loss_utils",
            "liger_kernel.transformers.model.output_classes",
            "liger_kernel.transformers.rope",
            "liger_kernel.transformers.auto_model",
            "liger_kernel.ops",
        )
        modules.update({s: _FlexMod(s) for s in l_subs})
        _noop_applier = lambda *a, **kw: None
        modules["liger_kernel.transformers.auto_model"].MODEL_TYPE_TO_APPLY_LIGER_FN = type(
            "_LigerMap", (dict,), {"get": lambda self, k, d=None: _noop_applier}
        )()

    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    return saved


def restore_modules(saved: dict[str, Any]) -> None:
    """Restore ``sys.modules`` entries replaced by :func:`install_cpu_gpu_stubs`."""
    for key, orig in saved.items():
        if orig is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = orig


def make_cpu_trainer_stub(
    cls_name: str,
    bounded_row_length_fn: Any,
    *,
    supports_cp: bool,
    pp_refusal: str | None,
    supports_lowp: bool,
    forward_rows: int,
    generation_shaped: bool,
) -> type:
    """Create a minimal trainer trait descriptor for CPU-only parallelism checks."""
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
                raise ValueError(f"{cls.__name__} does not support context parallelism (cp_size > 1).")
            if pp_size > 1 and cls.PP_REFUSAL is not None:
                raise ValueError(f"{cls.__name__} does not support pipeline parallelism: {cls.PP_REFUSAL}")

    _CPUTraitDescriptor.__name__ = cls_name
    _CPUTraitDescriptor.__qualname__ = cls_name
    return _CPUTraitDescriptor
