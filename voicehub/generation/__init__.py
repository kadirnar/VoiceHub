"""Native, reusable autoregressive generation primitives."""

from voicehub.generation.config import GenerationConfig
from voicehub.generation.ctc import (
    CTCDecodeResult,
    CTCTokenSpan,
    ctc_forced_alignment,
    ctc_greedy_decode,
    ctc_prefix_beam_search,
)
from voicehub.generation.engine import (
    AutoregressiveGenerator,
    DecoderStep,
    GenerationOutput,
    GenerationStepInput,
    GenerationStepOutput,
    LogitsProcessor,
)
from voicehub.generation.logits import (
    apply_repetition_penalty,
    filter_min_p,
    filter_top_k,
    filter_top_p,
    process_logits,
)
from voicehub.generation.sampling import create_generator, sample_next_token
from voicehub.generation.stopping import (
    EosStoppingCriterion,
    StoppingCriterion,
    evaluate_stopping_criteria,
    tokens_match_any,
)

__all__ = [
    "AutoregressiveGenerator",
    "CTCDecodeResult",
    "CTCTokenSpan",
    "DecoderStep",
    "EosStoppingCriterion",
    "GenerationConfig",
    "GenerationOutput",
    "GenerationStepInput",
    "GenerationStepOutput",
    "LogitsProcessor",
    "StoppingCriterion",
    "apply_repetition_penalty",
    "create_generator",
    "ctc_forced_alignment",
    "ctc_greedy_decode",
    "ctc_prefix_beam_search",
    "evaluate_stopping_criteria",
    "filter_min_p",
    "filter_top_k",
    "filter_top_p",
    "process_logits",
    "sample_next_token",
    "tokens_match_any",
]
