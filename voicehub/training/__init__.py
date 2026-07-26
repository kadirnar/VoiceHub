"""Model-family training adapters for every registered VoiceHub backend."""

from voicehub.training.adapters import (
    AcousticTrainingAdapter,
    BaseTrainingAdapter,
    CausalLMTrainingAdapter,
    CompositeTrainingAdapter,
    FlowMatchingTrainingAdapter,
    Seq2SeqTrainingAdapter,
)
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.collators import DataCollatorForTTSTraining
from voicehub.training.optimization import OptimizerBundle, SchedulerBundle
from voicehub.training.specs import (
    MODEL_TRAINING_SPECS,
    ModelTrainingSpec,
    TrainingFamily,
    get_training_spec,
    list_training_specs,
)

__all__ = [
    "AcousticTrainingAdapter",
    "AutoTrainingAdapter",
    "BaseTrainingAdapter",
    "CausalLMTrainingAdapter",
    "CompositeTrainingAdapter",
    "DataCollatorForTTSTraining",
    "FlowMatchingTrainingAdapter",
    "MODEL_TRAINING_SPECS",
    "ModelTrainingSpec",
    "OptimizerBundle",
    "Seq2SeqTrainingAdapter",
    "SchedulerBundle",
    "TrainingFamily",
    "get_training_spec",
    "list_training_specs",
]
