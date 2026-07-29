"""Native training objectives shared across speech model families."""

from voicehub.objectives.ctc import CTCLoss, ctc_loss
from voicehub.objectives.sequence import Seq2SeqCrossEntropyLoss, sequence_cross_entropy

__all__ = [
    "CTCLoss",
    "Seq2SeqCrossEntropyLoss",
    "ctc_loss",
    "sequence_cross_entropy",
]
