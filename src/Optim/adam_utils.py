"""Optim/adam_utils.py: Provides various utility functions for the Adam optimizer and its variants."""

import torch

# Above this many bytes (param + output), row ops use CPU staging to avoid GPU OOM spikes.
_GPU_COMPACT_BYTE_THRESHOLD = 1_500_000_000


def replace_param_group_data(optimizer: torch.optim.Optimizer, new_values: torch.Tensor, group_name: str, reset_state: bool = True) -> None:
    """Replaces the data of a parameter group with the given tensor."""
    for group in optimizer.param_groups:
        if group['name'] == group_name:
            if len(group['params']) != 1:
                raise NotImplementedError('"replace_param_group_data" only implemented for single-parameter groups.')
            param = group['params'][0]
            param.data = new_values
            if reset_state:
                state = optimizer.state[param]
                if state:
                    for val in ['exp_avg', 'exp_avg_sq']:
                        state[val].zero_()


def _maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _keep_indices(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=True)[0]


def _compact_rows(tensor: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    """Select rows by index; CPU-stages large GPU tensors to limit peak VRAM."""
    if keep_indices.numel() == 0:
        return tensor.new_empty((0, *tensor.shape[1:]))
    row_bytes = tensor[0].numel() * tensor.element_size() if tensor.numel() > 0 else tensor.element_size()
    out_bytes = keep_indices.numel() * row_bytes
    in_bytes = tensor.numel() * tensor.element_size()
    if tensor.is_cuda and in_bytes + out_bytes > _GPU_COMPACT_BYTE_THRESHOLD:
        idx_cpu = keep_indices.detach().cpu()
        kept_cpu = torch.index_select(tensor.detach().cpu(), 0, idx_cpu).contiguous()
        del idx_cpu
        _maybe_empty_cache()
        return kept_cpu.to(device=tensor.device, non_blocking=True)
    return torch.index_select(tensor, 0, keep_indices).contiguous()


def _concat_rows(base: torch.Tensor, extension: torch.Tensor) -> torch.Tensor:
    """Concatenate along dim 0; CPU-stages large GPU tensors to limit peak VRAM."""
    in_bytes = (base.numel() + extension.numel()) * base.element_size()
    out_bytes = (base.shape[0] + extension.shape[0]) * (
        base[0].numel() * base.element_size() if base.numel() > 0 else base.element_size()
    )
    if base.is_cuda and in_bytes + out_bytes > _GPU_COMPACT_BYTE_THRESHOLD:
        out_cpu = torch.cat((base.detach().cpu(), extension.detach().cpu()), dim=0).contiguous()
        del extension
        _maybe_empty_cache()
        return out_cpu.to(device=base.device, non_blocking=True)
    return torch.cat((base, extension), dim=0).contiguous()


def _replace_optimizer_param(
    optimizer: torch.optim.Optimizer,
    group: dict,
    new_param: torch.nn.Parameter,
    old_param: torch.nn.Parameter,
    state: dict | None,
) -> None:
    if state is not None:
        optimizer.state.pop(old_param, None)
        optimizer.state[new_param] = state
    group['params'][0] = new_param


def prune_param_groups(optimizer: torch.optim.Optimizer, mask: torch.Tensor, group_names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Removes parameter entries based on the given mask (True = keep)."""
    keep_indices = _keep_indices(mask)
    new_params = {}
    groups = [
        group for group in optimizer.param_groups
        if group_names is None or group['name'] in group_names
    ]
    groups.sort(key=lambda g: g['params'][0].numel())

    for group in groups:
        if len(group['params']) != 1:
            raise NotImplementedError('"prune_param_groups" only implemented for single-parameter groups.')
        old_param = group['params'][0]
        state = optimizer.state.get(old_param)
        new_param = torch.nn.Parameter(_compact_rows(old_param.data, keep_indices))
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                state[val] = _compact_rows(state[val], keep_indices)
        _replace_optimizer_param(optimizer, group, new_param, old_param, state)
        new_params[group['name']] = new_param
        _maybe_empty_cache()
    return new_params


def extend_param_groups(optimizer: torch.optim.Optimizer, additional_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extend existing parameters by concatenating the given tensors."""
    new_params = {}
    groups = sorted(
        [g for g in optimizer.param_groups if additional_params.get(g['name']) is not None],
        key=lambda g: g['params'][0].numel(),
    )
    for group in groups:
        if len(group['params']) != 1:
            raise NotImplementedError('"extend_param_groups" only implemented for single-parameter groups.')
        extension_tensor = additional_params[group['name']]
        old_param = group['params'][0]
        state = optimizer.state.get(old_param)
        new_param = torch.nn.Parameter(_concat_rows(old_param.data, extension_tensor))
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                state[val] = _concat_rows(state[val], torch.zeros_like(extension_tensor))
        _replace_optimizer_param(optimizer, group, new_param, old_param, state)
        new_params[group['name']] = new_param
        del extension_tensor
        _maybe_empty_cache()
    for group in optimizer.param_groups:
        if group['name'] not in new_params and len(group['params']) == 1:
            new_params[group['name']] = group['params'][0]
    return new_params


def reset_state(optimizer: torch.optim.Optimizer, group_names: list[str] | None = None, indices: torch.Tensor | None = None) -> None:
    """Resets the optimizer state for the specified parameter groups. If indices are provided, only those entries are reset."""
    for group in optimizer.param_groups:
        if group_names is not None and group['name'] not in group_names:
            continue
        if len(group['params']) != 1:
            raise NotImplementedError('"reset_state" only implemented for single-parameter groups.')
        param = group['params'][0]
        state = optimizer.state[param]
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                if indices is not None:
                    state[val][indices] = 0
                else:
                    state[val].zero_()


def sort_param_groups(optimizer: torch.optim.Optimizer, ordering: torch.Tensor, group_names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Sorts parameter entries based on the given ordering."""
    new_params = {}
    groups = [
        group for group in optimizer.param_groups
        if group_names is None or group['name'] in group_names
    ]
    groups.sort(key=lambda g: g['params'][0].numel())

    for group in groups:
        if len(group['params']) != 1:
            raise NotImplementedError('"sort_param_groups" only implemented for single-parameter groups.')
        old_param = group['params'][0]
        state = optimizer.state.get(old_param)
        new_param = torch.nn.Parameter(_compact_rows(old_param.data, ordering))
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                state[val] = _compact_rows(state[val], ordering)
        _replace_optimizer_param(optimizer, group, new_param, old_param, state)
        new_params[group['name']] = new_param
        _maybe_empty_cache()
    return new_params
