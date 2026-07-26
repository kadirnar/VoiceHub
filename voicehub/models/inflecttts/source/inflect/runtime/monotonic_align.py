"""Training-only alignment stub; deployable inference never calls maximum_path."""

def maximum_path(*args, **kwargs):
    raise RuntimeError("Monotonic alignment is unavailable in the inference package.")
