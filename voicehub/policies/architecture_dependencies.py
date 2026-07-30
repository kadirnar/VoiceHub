"""Import boundary for VoiceHub-owned architecture and training code.

The native runtime may use Python's standard library, VoiceHub itself,
and PyTorch as the tensor/autograd substrate.  Model frameworks,
upstream provider packages, optimization engines, and convenience DSP
libraries are forbidden in this layer.  They may only appear behind
explicit optional execution strategies outside the architecture
implementation.
"""

from __future__ import annotations

import ast
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

ALLOWED_NATIVE_IMPORT_ROOTS = frozenset({"torch", "voicehub"})
_DYNAMIC_IMPORT_INFRASTRUCTURE = frozenset({
    "auto.py",
    "automodel.py",
    "architectures/catalog.py",
    "architectures/specifications.py",
    "dependencies.py",
})
_PER_FILE_ALLOWED_IMPORT_ROOTS = {
    "integrations.py": frozenset({"wandb"}),
    "kernels/capabilities.py": frozenset({"cutlass", "triton"}),
    "kernels/cute_codecs.py": frozenset({"cutlass"}),
    "kernels/triton_activations.py": frozenset({"triton"}),
    "neural/backends/flash_attention4.py": frozenset({"flash_attn"}),
}
NATIVE_RUNTIME_DIRECTORIES = (
    "architectures",
    "audio.py",
    "base_model.py",
    "checkpointing",
    "components/neural/conformer",
    "components/audio/watermarking/wavmark",
    "components/audio/vocoders/vocos",
    "components/audio/codecs/dac/model",
    "components/audio/codecs/dac/nn",
    "components/audio/codecs/dac/utils",
    "components/audio/codecs/dac/compare",
    "components/audio/codecs/encodec",
    "data_collator.py",
    "generation",
    "neural",
    "objectives",
    "optimization",
    "processing",
    "streaming.py",
    "tokenization",
    "trainer.py",
    "trainer_utils.py",
    "training",
    "models/asr_hubert",
    "models/asr_moonshine",
    "models/asr_nemo",
    "models/asr_wenet",
    "models/asr_qwen3",
    "models/asr_granite_speech",
    "models/asr_parakeet_tdt",
    "models/asr_nemotron",
    "models/asr_cohere",
    "models/asr_seamless_m4t_v2",
    "models/asr_vibevoice",
    "models/asr_medasr",
    "models/asr_native/nemo.py",
    "models/asr_native/wenet.py",
    "models/asr_native/configuration.py",
    "models/asr_native/espnet.py",
    "models/asr_native/faster_whisper.py",
    "models/asr_native/funasr.py",
    "models/asr_native/openai_whisper.py",
    "models/asr_native/speechbrain.py",
    "models/asr_native/speechbrain_training.py",
    "models/asr_native/_wenet",
    "models/asr_native/whisper_compat.py",
    "models/asr_native/whisperx.py",
    "models/asr_tiron",
    "models/asr_transformers",
    "models/asr_transformers_multimodal",
    "models/asr_transformers_presets",
    "models/asr_wavlm",
    "models/asr_wav2vec2",
    "models/asr_whisper_native",
    "models/bark",
    "models/dia",
    "models/chatterbox/__init__.py",
    "models/chatterbox/configuration_chatterbox.py",
    "models/chatterbox/modeling_chatterbox.py",
    "models/chatterbox/inference.py",
    "models/chatterbox/tts.py",
    "models/chatterbox/vc.py",
    "models/chatterbox/checkpoint.py",
    "models/chatterbox/native_audio.py",
    "models/chatterbox/watermark.py",
    "models/chatterbox/training.py",
    "models/chatterbox/models",
    "models/csm/__init__.py",
    "models/csm/configuration_csm.py",
    "models/csm/modeling_csm.py",
    "models/csm/inference.py",
    "models/csm/training.py",
    "models/csm/source/moshi/models/__init__.py",
    "models/csm/source/moshi/models/compression.py",
    "models/csm/source/moshi/modules",
    "models/csm/source/moshi/quantization",
    "models/csm/source/moshi/utils/__init__.py",
    "models/csm/source/moshi/utils/compile.py",
    "models/csm/source/moshi/utils/quantize.py",
    "models/conversationtts/__init__.py",
    "models/conversationtts/configuration_conversationtts.py",
    "models/conversationtts/inference.py",
    "models/conversationtts/modeling_conversationtts.py",
    "models/conversationtts/runtime.py",
    "models/conversationtts/source/conversationtts/inference/generator.py",
    "models/conversationtts/source/conversationtts/models/model_new.py",
    "models/conversationtts/source/conversationtts/tools/tokenizer/abs_tokenizer.py",
    "models/conversationtts/source/conversationtts/tools/tokenizer/common.py",
    "models/conversationtts/source/conversationtts/tools/tokenizer/Text2ID/text_tokenizer.py",
    "models/conversationtts/source/conversationtts/tools/tokenizer/MimiCodec",
    "models/echo",
    "models/f5tts/__init__.py",
    "models/f5tts/configuration_f5tts.py",
    "models/f5tts/inference.py",
    "models/f5tts/modeling_f5tts.py",
    "models/fishtts/__init__.py",
    "models/fishtts/configuration_fishtts.py",
    "models/fishtts/inference.py",
    "models/fishtts/modeling_fishtts.py",
    "models/fishtts/training.py",
    "models/gptsovits/__init__.py",
    "models/gptsovits/configuration_gptsovits.py",
    "models/gptsovits/modeling_gptsovits.py",
    "models/gptsovits/inference.py",
    "models/gptsovits/training.py",
    "models/mosstts/__init__.py",
    "models/mosstts/configuration_mosstts.py",
    "models/mosstts/modeling_mosstts.py",
    "models/mosstts/inference.py",
    "models/mosstts/training.py",
    "models/higgstts/__init__.py",
    "models/higgstts/configuration_higgstts.py",
    "models/higgstts/inference.py",
    "models/higgstts/modeling_higgstts.py",
    "models/higgstts/training.py",
    "models/inflecttts/__init__.py",
    "models/inflecttts/configuration_inflecttts.py",
    "models/inflecttts/inference.py",
    "models/inflecttts/modeling_inflecttts.py",
    "models/inflecttts/training.py",
    "models/irodoritts/__init__.py",
    "models/irodoritts/configuration_irodoritts.py",
    "models/irodoritts/inference.py",
    "models/irodoritts/modeling_irodoritts.py",
    "models/irodoritts/training.py",
    "models/cosyvoice/__init__.py",
    "models/cosyvoice/configuration_cosyvoice.py",
    "models/cosyvoice/inference.py",
    "models/cosyvoice/modeling_cosyvoice.py",
    "models/cosyvoice/training.py",
    "models/cosyvoice_native",
    "models/xtts/__init__.py",
    "models/xtts/configuration_xtts.py",
    "models/xtts/inference.py",
    "models/xtts/modeling_xtts.py",
    "models/xtts/training.py",
    "models/xtts_native",
    "models/kokoro",
    "models/llasa/__init__.py",
    "models/llasa/artifacts.py",
    "models/llasa/checkpoint.py",
    "models/llasa/configuration_llasa.py",
    "models/llasa/inference.py",
    "models/llasa/modeling_llasa.py",
    "models/llasa/tokenization_llasa.py",
    "models/llasa/training.py",
    "models/llasa/xcodec2.py",
    "models/melotts/__init__.py",
    "models/melotts/configuration_melotts.py",
    "models/melotts/inference.py",
    "models/melotts/modeling_melotts.py",
    "models/melotts/training.py",
    "models/melotts/source/melo/models.py",
    "models/melotts/source/melo/modules.py",
    "models/melotts/source/melo/attentions.py",
    "models/melotts/source/melo/commons.py",
    "models/melotts/source/melo/transforms.py",
    "models/melotts/source/melo/monotonic_align",
    "models/openvoice/__init__.py",
    "models/openvoice/configuration_openvoice.py",
    "models/openvoice/modeling_openvoice.py",
    "models/openvoice/inference.py",
    "models/openvoice/training.py",
    "models/openvoice/source/openvoice/models.py",
    "models/openvoice/source/openvoice/modules.py",
    "models/openvoice/source/openvoice/commons.py",
    "models/openvoice/source/openvoice/attentions.py",
    "models/openvoice/source/openvoice/transforms.py",
    "models/neutts/__init__.py",
    "models/neutts/configuration_neutts.py",
    "models/neutts/modeling_neutts.py",
    "models/neutts/inference.py",
    "models/neutts/training.py",
    "models/outetts/__init__.py",
    "models/outetts/configuration_outetts.py",
    "models/outetts/inference.py",
    "models/outetts/modeling_outetts.py",
    "models/outetts/training.py",
    "models/orpheustts",
    "models/parlertts/__init__.py",
    "models/parlertts/configuration_parlertts.py",
    "models/parlertts/modeling_parlertts.py",
    "models/parlertts/inference.py",
    "models/parlertts/training.py",
    "models/qwen3tts/__init__.py",
    "models/qwen3tts/configuration_qwen3tts.py",
    "models/qwen3tts/modeling_qwen3tts.py",
    "models/qwen3tts/inference.py",
    "models/qwen3tts/training.py",
    "models/speecht5",
    "models/supertonic/__init__.py",
    "models/supertonic/configuration_supertonic.py",
    "models/supertonic/modeling_supertonic.py",
    "models/supertonic/inference.py",
    "models/supertonic/training.py",
    "models/styletts2/__init__.py",
    "models/styletts2/configuration_styletts2.py",
    "models/styletts2/modeling_styletts2.py",
    "models/styletts2/inference.py",
    "models/styletts2/runtime.py",
    "models/styletts2/training.py",
    "models/styletts2/monotonic_align.py",
    "models/styletts2/source/styletts2/models.py",
    "models/styletts2/source/styletts2/Modules/hifigan.py",
    "models/styletts2/source/styletts2/Modules/istftnet.py",
    "models/styletts2/source/styletts2/Modules/discriminators.py",
    "models/styletts2/source/styletts2/Modules/utils.py",
    "models/styletts2/source/styletts2/Modules/diffusion",
    "models/vits",
    "models/vui",
    "models/vibevoice/__init__.py",
    "models/vibevoice/configuration_vibevoice.py",
    "models/vibevoice/inference.py",
    "models/vibevoice/modeling_vibevoice.py",
    "models/vibevoice/training.py",
    "models/voxcpm/__init__.py",
    "models/voxcpm/configuration_voxcpm.py",
    "models/voxcpm/inference.py",
    "models/voxcpm/modeling_voxcpm.py",
    "models/voxcpm/training.py",
    "models/voxcpm_native",
    "models/omnivoice/__init__.py",
    "models/omnivoice/configuration_omnivoice.py",
    "models/omnivoice/modeling_omnivoice.py",
    "models/omnivoice/inference.py",
    "models/omnivoice/training.py",
    "models/omnivoice_native",
    "models/zonos/__init__.py",
    "models/zonos/configuration_zonos.py",
    "models/zonos/inference.py",
    "models/zonos/modeling_zonos.py",
    "models/zonos/training.py",
    "models/zonos2/__init__.py",
    "models/zonos2/configuration_zonos2.py",
    "models/zonos2/inference.py",
    "models/zonos2/modeling_zonos2.py",
    "models/zonos2/training.py",
    "models/vad_auditok",
    "models/vad_funasr",
    "models/vad_nemo",
    "models/vad_sherpa_onnx",
    "models/vad_silero",
    "models/vad_transformers",
    "models/vad_webrtc",
    "models/vad_pyannote",
    "models/vad_pyannote_segmentation",
    "models/vad_pyannote_brouhaha",
    "models/vad_speechbrain",
)


@dataclass(frozen=True, order=True)
class ImportPolicyViolation:
    """One external import found inside the native runtime boundary."""

    path: Path
    line: int
    module: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: external import {self.module!r}"


def _module_root(module: str) -> str:
    return module.partition(".")[0]


def _is_allowed(module: str, *, allowed_roots: frozenset[str]) -> bool:
    root = _module_root(module)
    return (root in allowed_roots or root in sys.stdlib_module_names or root == "__future__")


def _literal_dynamic_import(node: ast.Call) -> tuple[str, int] | None:
    function_name = None
    if isinstance(node.func, ast.Name):
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        function_name = node.func.attr
    if function_name not in {"__import__", "import_module", "import_optional"}:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    module = node.args[0].value
    if not isinstance(module, str) or not module:
        return None
    return module, node.lineno


def _dynamic_import_function_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        function_name = node.func.attr
    else:
        return None
    if function_name in {"__import__", "import_module", "import_optional"}:
        return function_name
    return None


def _parse_source(path: Path) -> ast.Module:
    try:
        return ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"Could not inspect Python source {path}: {error}.") from error


class _LazyNamespaceImportVisitor(ast.NodeVisitor):
    """Recognize unresolved imports used only by a lazy package namespace."""

    def __init__(self) -> None:
        self._function_stack: list[str] = []
        self.unresolved_imports: list[bool] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if (_dynamic_import_function_name(node) is not None and _literal_dynamic_import(node) is None):
            self.unresolved_imports.append(
                bool(self._function_stack) and self._function_stack[-1] == "__getattr__", )
        self.generic_visit(node)


def _is_lazy_namespace_initializer(path: Path) -> bool:
    """Return whether unresolved imports are confined to package
    ``__getattr__``."""
    if path.name != "__init__.py":
        return False
    visitor = _LazyNamespaceImportVisitor()
    visitor.visit(_parse_source(path))
    return bool(visitor.unresolved_imports) and all(visitor.unresolved_imports)


def inspect_native_imports(
    path: str | Path,
    *,
    allowed_roots: Iterable[str] = ALLOWED_NATIVE_IMPORT_ROOTS,
    allow_unresolved_dynamic_imports: bool = False,
) -> tuple[ImportPolicyViolation, ...]:
    """Inspect one Python file without importing it."""
    source_path = Path(path)
    normalized_roots = frozenset(allowed_roots)
    tree = _parse_source(source_path)
    violations: set[ImportPolicyViolation] = set()
    for node in ast.walk(tree):
        modules: tuple[tuple[str, int], ...] = ()
        if isinstance(node, ast.Import):
            modules = tuple((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = ((node.module, node.lineno), )
        elif isinstance(node, ast.Call):
            dynamic = _literal_dynamic_import(node)
            if dynamic is not None:
                modules = (dynamic, )
            elif not allow_unresolved_dynamic_imports:
                function_name = _dynamic_import_function_name(node)
                if function_name is not None:
                    modules = ((f"<dynamic:{function_name}>", node.lineno), )
        for module, line in modules:
            if not _is_allowed(module, allowed_roots=normalized_roots):
                violations.add(ImportPolicyViolation(
                    path=source_path,
                    line=line,
                    module=module,
                ))
    return tuple(sorted(violations))


def _module_name_for_path(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root)
    if relative.name == "__init__.py":
        parts = relative.parent.parts
    else:
        parts = relative.with_suffix("").parts
    return ".".join((package_root.name, *parts))


def _resolve_internal_module_path(
    package_root: Path,
    module_name: str,
) -> Path | None:
    package_name = package_root.name
    if module_name == package_name:
        initializer = package_root / "__init__.py"
        return initializer if initializer.is_file() else None
    prefix = f"{package_name}."
    if not module_name.startswith(prefix):
        return None
    relative_parts = module_name[len(prefix):].split(".")
    module_path = package_root.joinpath(*relative_parts).with_suffix(".py")
    if module_path.is_file():
        return module_path
    initializer = package_root.joinpath(*relative_parts, "__init__.py")
    return initializer if initializer.is_file() else None


def _absolute_import_name(
    *,
    imported_name: str,
    level: int,
    package_name: str,
) -> str | None:
    if level == 0:
        return imported_name
    relative_name = f"{'.' * level}{imported_name}"
    try:
        return resolve_name(relative_name, package_name)
    except (ImportError, ValueError):
        return None


def _iter_internal_import_names(
    tree: ast.Module,
    *,
    module_name: str,
    is_package: bool,
    package_root_name: str,
) -> Iterable[str]:
    package_name = module_name if is_package else module_name.rpartition(".")[0]
    internal_prefix = f"{package_root_name}."

    def is_internal(name: str) -> bool:
        return name == package_root_name or name.startswith(internal_prefix)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_internal(alias.name):
                    yield alias.name
            continue
        if isinstance(node, ast.ImportFrom):
            imported_name = _absolute_import_name(
                imported_name=node.module or "",
                level=node.level,
                package_name=package_name,
            )
            if imported_name is None or not is_internal(imported_name):
                continue
            yield imported_name
            for alias in node.names:
                if alias.name != "*":
                    yield f"{imported_name}.{alias.name}"
            continue
        if not isinstance(node, ast.Call):
            continue
        dynamic = _literal_dynamic_import(node)
        if dynamic is None:
            continue
        imported_name = dynamic[0]
        if imported_name.startswith("."):
            imported_name = _absolute_import_name(
                imported_name=imported_name.lstrip("."),
                level=len(imported_name) - len(imported_name.lstrip(".")),
                package_name=package_name,
            )
        if imported_name is not None and is_internal(imported_name):
            yield imported_name


def inspect_native_runtime(
    package_root: str | Path,
    *,
    directories: Iterable[str] = NATIVE_RUNTIME_DIRECTORIES,
    allowed_roots: Iterable[str] = ALLOWED_NATIVE_IMPORT_ROOTS,
) -> tuple[ImportPolicyViolation, ...]:
    """Inspect all present native runtime files and their package initializers.

    Importing ``voicehub.a.b.module`` executes every package
    ``__init__.py`` between ``voicehub`` and ``module``.  Auditing only
    ``module.py`` would therefore allow an eager dependency in an
    ancestor package to bypass the native boundary.
    """
    violations: list[ImportPolicyViolation] = []
    root = Path(package_root)
    for path in collect_native_import_closure(
            package_root,
            directories=directories,
    ):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = ""
        allow_dynamic = (relative in _DYNAMIC_IMPORT_INFRASTRUCTURE or _is_lazy_namespace_initializer(path))
        path_allowed_roots = (
            frozenset(allowed_roots)
            | _PER_FILE_ALLOWED_IMPORT_ROOTS.get(relative, frozenset()))
        violations.extend(
            inspect_native_imports(
                path,
                allowed_roots=path_allowed_roots,
                allow_unresolved_dynamic_imports=allow_dynamic,
            ))
    return tuple(sorted(violations))


def collect_native_runtime_paths(
    package_root: str | Path,
    *,
    directories: Iterable[str] = NATIVE_RUNTIME_DIRECTORIES,
) -> tuple[Path, ...]:
    """Resolve the complete, auditable file set for a native boundary."""
    root = Path(package_root)
    root_init = root / "__init__.py"
    paths: set[Path] = set()

    def add_package_initializers(path: Path) -> None:
        parent = path.parent
        if root_init.is_file():
            paths.add(root_init)
        try:
            relative_parent = parent.relative_to(root)
        except ValueError:
            return
        current = root
        for part in relative_parent.parts:
            current /= part
            initializer = current / "__init__.py"
            if initializer.is_file():
                paths.add(initializer)

    for directory in directories:
        runtime_path = root / directory
        if runtime_path.is_file():
            runtime_files = (runtime_path, )
        elif runtime_path.is_dir():
            runtime_files = tuple(sorted(runtime_path.rglob("*.py")))
        else:
            continue
        for path in runtime_files:
            add_package_initializers(path)
            paths.add(path)
    return tuple(sorted(paths))


def collect_native_import_closure(
    package_root: str | Path,
    *,
    directories: Iterable[str] = NATIVE_RUNTIME_DIRECTORIES,
) -> tuple[Path, ...]:
    """Resolve the fixed-point VoiceHub import closure of the native boundary.

    The explicit native paths are seeds, not an exemption list. Every
    statically discoverable internal import is recursively inspected,
    including relative imports, package initializers, and literal
    ``import_module``/``__import__``/``import_optional`` calls.
    """
    root = Path(package_root)
    root_init = root / "__init__.py"
    paths = set(collect_native_runtime_paths(
        root,
        directories=directories,
    ))
    pending = deque(sorted(paths))

    def add_path(path: Path) -> None:
        if path not in paths:
            paths.add(path)
            pending.append(path)

        if root_init.is_file() and root_init not in paths:
            paths.add(root_init)
            pending.append(root_init)
        try:
            relative_parent = path.parent.relative_to(root)
        except ValueError:
            return
        current = root
        for part in relative_parent.parts:
            current /= part
            initializer = current / "__init__.py"
            if initializer.is_file() and initializer not in paths:
                paths.add(initializer)
                pending.append(initializer)

    while pending:
        path = pending.popleft()
        module_name = _module_name_for_path(root, path)
        tree = _parse_source(path)
        for imported_name in _iter_internal_import_names(
                tree,
                module_name=module_name,
                is_package=path.name == "__init__.py",
                package_root_name=root.name,
        ):
            imported_path = _resolve_internal_module_path(root, imported_name)
            if imported_path is not None:
                add_path(imported_path)

    return tuple(sorted(paths))


def require_native_runtime_independence(
    package_root: str | Path,
    *,
    directories: Iterable[str] = NATIVE_RUNTIME_DIRECTORIES,
    allowed_roots: Iterable[str] = ALLOWED_NATIVE_IMPORT_ROOTS,
) -> None:
    """Raise with every violation instead of failing on only the first."""
    violations = inspect_native_runtime(
        package_root,
        directories=directories,
        allowed_roots=allowed_roots,
    )
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise RuntimeError("VoiceHub native runtime imports external architecture code:\n"
                           f"{details}")


__all__ = [
    "ALLOWED_NATIVE_IMPORT_ROOTS",
    "NATIVE_RUNTIME_DIRECTORIES",
    "ImportPolicyViolation",
    "collect_native_import_closure",
    "collect_native_runtime_paths",
    "inspect_native_imports",
    "inspect_native_runtime",
    "require_native_runtime_independence",
]
