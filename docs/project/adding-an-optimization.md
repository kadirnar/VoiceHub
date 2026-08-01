---
description: Add one optimization pass that is immediately available to every speech model.
---

# Add an optimization

An optimization is one class plus one registry call. Every model that inherits
the VoiceHub speech base classes sees the new pass automatically. The pass
checks the loaded runtime before it changes anything.

## Implement the pass

```python
from voicehub.optimization import (
    OptimizationCapabilities,
    OptimizationMode,
    OptimizationPass,
    OptimizationCompatibilityError,
    PassResult,
    register_optimization_pass,
)


@register_optimization_pass("acme-eval-mode")
class AcmeEvalModePass(OptimizationPass):
    pass_id = "acme.eval-mode"
    pass_version = "1"
    capabilities = OptimizationCapabilities(
        modes=(OptimizationMode.INFERENCE,),
        reversible=True,
    )

    def manifest_configuration(self):
        return {}

    def validate(self, model, context):
        super().validate(model, context)
        if not callable(getattr(model, "eval", None)):
            raise OptimizationCompatibilityError("model has no eval() method")

    def apply(self, model, context):
        was_training = bool(getattr(model, "training", False))
        model.eval()
        return PassResult(model=model, state={"was_training": was_training})

    def restore(self, model, state, context):
        model.train(state["was_training"])
        return model
```

A class is callable, so the decorator stores it as a lazy factory. A function
that returns a configured pass works too.

## Apply it to any task

```python
from voicehub import AutoModel

model = AutoModel.from_pretrained(checkpoint, model_type=model_type)
result = model.apply_optimization_plan("acme-eval-mode", mode="inference")
print(result.manifest())
```

The same method exists on TTS, ASR, and VAD wrappers. Registration means the
pass can be requested everywhere; `validate()` decides whether one concrete
runtime is compatible. All validations finish before the first pass mutates
the model, and earlier reversible passes roll back if a later pass fails.

## Declare capabilities honestly

`OptimizationCapabilities` describes execution constraints:

- `modes`: inference, training, or both;
- `devices` and `dtypes`: supported runtime values;
- `streaming_safe` and `distributed_safe`: concurrency guarantees;
- `persistent`: whether transformed state may be checkpointed;
- `reversible`: whether `restore()` is implemented;
- topology flags: whether parameter names or structure change.

Set `requires_architecture_support = True` only when the pass relies on a
manually audited architecture contract that runtime inspection cannot prove.
Most extension passes should validate a protocol or module surface directly,
which avoids editing every model when the pass is added.

## Test four paths

Test that:

1. registration is lazy;
2. validation rejects an incompatible runtime before mutation;
3. application produces deterministic manifest metadata;
4. restoration returns the original runtime and state keys.

Use an isolated `OptimizationPassRegistry` in unit tests when global
registration is unnecessary. See [Library architecture](../concepts/architecture.md)
for transaction and lifecycle details.
