"""Registration and explicit loading for optional PyTorch CUDA extensions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from voicehub.kernels.capabilities import cuda_extension_capability

_EXTENSION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPERATOR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*$")


class CudaExtensionError(RuntimeError):
    """Base error for CUDA-extension declaration or loading."""


class CudaExtensionUnavailableError(CudaExtensionError):
    """The local runtime cannot build and execute a CUDA extension."""


class CudaExtensionBuildError(CudaExtensionError):
    """PyTorch failed while compiling or loading a CUDA extension."""


@dataclass(frozen=True)
class CudaExtensionSpec:
    """Declarative source and dispatcher contract for one extension."""

    name: str
    sources: tuple[str | Path, ...]
    operators: tuple[str, ...] = ()
    extra_cflags: tuple[str, ...] = ("-O3", )
    extra_cuda_cflags: tuple[str, ...] = ("-O3", )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _EXTENSION_NAME.fullmatch(self.name) is None:
            raise ValueError("CUDA extension names must be valid Python/C++ identifiers.")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("CUDA extensions require a non-empty source tuple.")
        sources = tuple(Path(source).expanduser().resolve() for source in self.sources)
        if not any(source.suffix == ".cu" for source in sources):
            raise ValueError("CUDA extensions require at least one `.cu` source.")
        object.__setattr__(self, "sources", sources)

        if not isinstance(self.operators, tuple):
            raise TypeError("CUDA extension operators must be a tuple.")
        if len(self.operators) != len(set(self.operators)):
            raise ValueError("CUDA extension operators cannot contain duplicates.")
        if any(not isinstance(operator, str) or _OPERATOR_NAME.fullmatch(operator) is None
               for operator in self.operators):
            raise ValueError("CUDA operators must use the `namespace::operator` form.")
        for name in ("extra_cflags", "extra_cuda_cflags"):
            flags = getattr(self, name)
            if not isinstance(flags, tuple) or any(not isinstance(flag, str) or not flag.strip()
                                                   for flag in flags):
                raise TypeError(f"`{name}` must be a tuple of non-empty strings.")


@dataclass(frozen=True)
class LoadedCudaExtension:
    """Successful extension load retained to prevent duplicate registration."""

    spec: CudaExtensionSpec
    module: Any


def _compile_extension(
    spec: CudaExtensionSpec,
    *,
    build_directory: str | None,
    verbose: bool,
) -> Any:
    from torch.utils.cpp_extension import load

    return load(
        name=spec.name,
        sources=[str(source) for source in spec.sources],
        extra_cflags=list(spec.extra_cflags),
        extra_cuda_cflags=list(spec.extra_cuda_cflags),
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=True,
        is_python_module=False,
        keep_intermediates=False,
    )


def _resolve_operator(qualified_name: str) -> Any:
    namespace, operator = qualified_name.split("::", 1)
    try:
        return getattr(getattr(torch.ops, namespace), operator)
    except AttributeError as error:
        raise CudaExtensionBuildError(f"CUDA extension did not register {qualified_name!r}.") from error


class CudaExtensionRegistry:
    """Thread-safe source registry with a deliberately explicit build step."""

    def __init__(self) -> None:
        self._specs: dict[str, CudaExtensionSpec] = {}
        self._loaded: dict[str, LoadedCudaExtension] = {}
        self._lock = RLock()

    def register(
        self,
        spec: CudaExtensionSpec,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(spec, CudaExtensionSpec):
            raise TypeError("`spec` must be a CudaExtensionSpec.")
        if not isinstance(replace, bool):
            raise TypeError("`replace` must be a boolean.")
        with self._lock:
            if spec.name in self._loaded:
                if self._loaded[spec.name].spec == spec:
                    return
                raise ValueError(f"Loaded CUDA extension {spec.name!r} cannot be replaced.")
            if spec.name in self._specs and not replace:
                if self._specs[spec.name] == spec:
                    return
                raise ValueError(f"CUDA extension {spec.name!r} is already registered.")
            self._specs[spec.name] = spec

    def get(self, name: str) -> CudaExtensionSpec:
        if not isinstance(name, str):
            raise TypeError("CUDA extension names must be strings.")
        with self._lock:
            try:
                return self._specs[name]
            except KeyError as error:
                raise KeyError(
                    f"Unknown CUDA extension {name!r}. Available: "
                    f"{', '.join(sorted(self._specs)) or 'none'}.") from error

    def list(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._specs))

    def is_loaded(self, name: str) -> bool:
        if not isinstance(name, str):
            raise TypeError("CUDA extension names must be strings.")
        with self._lock:
            return name in self._loaded

    def load(
        self,
        name: str,
        *,
        build_directory: str | Path | None = None,
        verbose: bool = False,
        require_runtime: bool = True,
    ) -> LoadedCudaExtension:
        """Compile and load a registered extension on explicit user request.

        Importing :mod:`voicehub.kernels` never calls this method.
        PyTorch's extension cache handles rebuild avoidance across
        processes; this registry additionally prevents duplicate
        dispatcher registration in the current process.
        """
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be a boolean.")
        if not isinstance(require_runtime, bool):
            raise TypeError("`require_runtime` must be a boolean.")
        resolved_build_directory = None
        if build_directory is not None:
            path = Path(build_directory).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"CUDA build directory does not exist: {str(path)!r}.")
            resolved_build_directory = str(path)

        with self._lock:
            loaded = self._loaded.get(name)
            if loaded is not None:
                return loaded
            spec = self.get(name)
            missing_sources = tuple(str(source) for source in spec.sources if not source.is_file())
            if missing_sources:
                raise CudaExtensionBuildError(
                    "CUDA extension sources are missing: "
                    f"{', '.join(missing_sources)}.")
            capability = cuda_extension_capability(require_runtime=require_runtime, )
            if not capability.available:
                raise CudaExtensionUnavailableError(capability.reason)
            try:
                module = _compile_extension(
                    spec,
                    build_directory=resolved_build_directory,
                    verbose=verbose,
                )
            except Exception as error:
                raise CudaExtensionBuildError(f"Could not build CUDA extension {name!r}: {error}") from error
            for operator in spec.operators:
                _resolve_operator(operator)
            loaded = LoadedCudaExtension(spec=spec, module=module)
            self._loaded[name] = loaded
            return loaded


CUDA_EXTENSIONS = CudaExtensionRegistry()


def register_cuda_extension(
    spec: CudaExtensionSpec,
    *,
    replace: bool = False,
) -> None:
    """Register a source extension without importing a compiler backend."""
    CUDA_EXTENSIONS.register(spec, replace=replace)


def load_cuda_extension(
    name: str,
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
    require_runtime: bool = True,
) -> LoadedCudaExtension:
    """Explicitly compile and load a process-wide CUDA extension."""
    return CUDA_EXTENSIONS.load(
        name,
        build_directory=build_directory,
        verbose=verbose,
        require_runtime=require_runtime,
    )
