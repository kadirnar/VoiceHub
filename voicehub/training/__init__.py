"""Model-family training adapters for every registered VoiceHub backend."""

from voicehub.training.adapters import (
    AcousticTrainingAdapter,
    BaseTrainingAdapter,
    CausalLMTrainingAdapter,
    CompositeTrainingAdapter,
    FlowMatchingTrainingAdapter,
    Seq2SeqTrainingAdapter,
    VITSTrainingAdapter,
)
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.collators import DataCollatorForTTSTraining, TTSFieldSchema
from voicehub.training.contracts import (
    TrainingContext,
    TrainingPhaseKind,
    TrainingPhaseSpec,
    TrainingRecipeKind,
    TrainingSupport,
)
from voicehub.training.optimization import OptimizerBundle, SchedulerBundle
from voicehub.training.specs import (
    MODEL_TRAINING_SPECS,
    ModelTrainingSpec,
    TrainingFamily,
    get_training_spec,
    list_training_specs,
    register_training_alias,
    register_training_spec,
    unregister_training_alias,
    unregister_training_spec,
)
from voicehub.training.strategy import (
    TorchTrainingStrategy,
    TrainingStrategy,
    get_training_strategy,
    list_training_strategies,
    register_training_strategy,
    unregister_training_strategy,
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
    "TrainingContext",
    "TrainingPhaseKind",
    "TrainingPhaseSpec",
    "TrainingRecipeKind",
    "TrainingStrategy",
    "TrainingSupport",
    "TTSFieldSchema",
    "TorchTrainingStrategy",
    "VITSTrainingAdapter",
    "get_training_strategy",
    "get_training_spec",
    "list_training_strategies",
    "list_training_specs",
    "register_training_alias",
    "register_training_spec",
    "register_training_strategy",
    "unregister_training_alias",
    "unregister_training_spec",
    "unregister_training_strategy",
]
