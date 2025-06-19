def load_what_you_can(checkpoint: dict, model):
    """
    This method takes a checkpoint and loads as many weights from it as possible:

    If they are the same shape, there's nothing to do

    Will load the smallest shape otherwise.
    """
    import torch

    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint

    for name, param in checkpoint_state_dict.items():
        if name not in model_state_dict:
            print(f"Ignoring parameter '{name}' because it is not found in the model")
            continue

        model_state = model_state_dict[name]
        mshape = model_state.shape
        pshape = param.shape

        if pshape == mshape:
            model_state.copy_(param)
            continue

        if len(pshape) != len(mshape):
            # Completely different shapes so probably unwise to merge
            continue

        min_shape = [min(param.shape[i], model_state.shape[i]) for i in range(len(param.shape))]
        print(name, "model:", mshape, "chkpt:", pshape, "loading:", min_shape)
        idxs = torch.meshgrid(*[torch.arange(s) for s in min_shape])
        model_state[tuple(idxs)].copy_(param[tuple(idxs)])

    return model.load_state_dict(model_state_dict)


def decompile_state_dict(state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # state_dict = convert_old_weight_norm_to_new(state_dict)
    return {k.replace("module.", ""): v for k, v in state_dict.items()}
