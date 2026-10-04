"""Persistent product-space local update and rank retraction."""

import math
from collections import OrderedDict
import torch
from federatedscope.barylora.tools import activity_from_second_moment

TANGENT_PROJECTION_MODES = ("orthogonal",)
GRAD_DAMPING_MODES = ("relative", "absolute")
RANK_RETURN_SVD_MODES = ("exact", "randomized")
_MIN_GRAM_SHIFT = 1e-12


def _cfg_value(cfg, name, default):
    return getattr(getattr(cfg, "lora", cfg), name, default)


def _lora_parameter_pairs(model):
    params = OrderedDict(model.named_parameters())
    for a_name, a_param in params.items():
        if "lora_A" not in a_name:
            continue
        b_name = a_name.replace("lora_A", "lora_B")
        if b_name not in params:
            raise ValueError(f"Missing matching LoRA B parameter for {a_name}")
        b_param = params[b_name]
        if a_param.ndim != 2 or b_param.ndim != 2:
            raise ValueError(
                f"LoRA parameters must be 2D, got {a_name}{tuple(a_param.shape)} and {b_name}{tuple(b_param.shape)}"
            )
        if b_param.shape[1] != a_param.shape[0]:
            raise ValueError(
                f"LoRA rank mismatch for {a_name}/{b_name}: A rank {a_param.shape[0]}, B rank {b_param.shape[1]}"
            )
        yield (a_name, a_param, b_name, b_param)


def _solve_damped(gram, rhs, damping):
    rank = gram.shape[0]
    eye = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + damping * eye, rhs)


def _gram_damping(gram, mode, absolute, relative):
    """Scale the Gram shift; the absolute floor handles zero-initialized B."""
    if mode == "relative":
        if relative <= 0.0:
            return 0.0
        spectral = float(torch.linalg.matrix_norm(gram, ord=2))
        if spectral > 0.0 and math.isfinite(spectral):
            return relative * spectral
        return max(float(absolute), _MIN_GRAM_SHIFT)
    if mode == "absolute":
        return max(float(absolute), 0.0)
    raise ValueError(
        f"cfg.lora.bary_damping_mode must be one of {list(GRAD_DAMPING_MODES)}, got {mode!r}."
    )


def _recover_effective_update_direction(
    a,
    b,
    grad_a,
    grad_b,
    damping,
    damping_mode="relative",
    damping_rel=1e-07,
    projection="orthogonal",
):
    """Minimum-norm tangent direction: P_B G + G P_A - P_B G P_A."""
    if projection not in TANGENT_PROJECTION_MODES:
        raise ValueError(
            f"cfg.lora.bary_direction_projection must be one of {list(TANGENT_PROJECTION_MODES)}, got {projection!r}."
        )
    a32 = a.detach().to(dtype=torch.float32)
    b32 = b.detach().to(dtype=torch.float32)
    grad_a32 = grad_a.detach().to(dtype=torch.float32)
    grad_b32 = grad_b.detach().to(dtype=torch.float32)
    gram_b = b32.T @ b32
    gram_a = a32 @ a32.T
    damping_b = _gram_damping(gram_b, damping_mode, damping, damping_rel)
    damping_a = _gram_damping(gram_a, damping_mode, damping, damping_rel)
    grad_mu_from_b = b32 @ _solve_damped(gram_b, grad_a32, damping_b)
    grad_mu_from_a = grad_b32 @ _solve_damped(gram_a, a32, damping_a)
    overlap = b32 @ _solve_damped(gram_b, b32.T @ grad_mu_from_a, damping_b)
    return grad_mu_from_b + grad_mu_from_a - overlap


def _rank_project_on_device(
    mu, rank, a_dtype, b_dtype, mode="exact", randomized_q=0, randomized_niter=2
):
    """Balanced rank-r factors, using the configured exact/randomized SVD."""
    if mode not in RANK_RETURN_SVD_MODES:
        raise ValueError(
            f"cfg.lora.bary_local_rank_return must be one of {list(RANK_RETURN_SVD_MODES)}, got {mode!r}."
        )
    mu32 = mu.to(dtype=torch.float32)
    if mode == "exact":
        u, s, vh = torch.linalg.svd(mu32, full_matrices=False)
        keep = min(rank, s.numel())
        sqrt_s = torch.sqrt(torch.clamp(s[:keep], min=0.0))
        b_new = u[:, :keep] * sqrt_s.unsqueeze(0)
        a_new = sqrt_s.unsqueeze(1) * vh[:keep, :]
    else:
        q = int(randomized_q) if int(randomized_q) > 0 else int(rank) + 16
        q = max(1, min(q, int(min(mu32.shape))))
        u, s, v = torch.svd_lowrank(mu32, q=q, niter=int(randomized_niter))
        keep = min(rank, s.numel())
        sqrt_s = torch.sqrt(torch.clamp(s[:keep], min=0.0))
        b_new = u[:, :keep] * sqrt_s.unsqueeze(0)
        a_new = sqrt_s.unsqueeze(1) * v[:, :keep].T
    if keep < rank:
        a_pad = torch.zeros(
            rank - keep, a_new.shape[1], device=mu.device, dtype=a_new.dtype
        )
        b_pad = torch.zeros(
            b_new.shape[0], rank - keep, device=mu.device, dtype=b_new.dtype
        )
        a_new = torch.cat([a_new, a_pad], dim=0)
        b_new = torch.cat([b_new, b_pad], dim=1)
    return (a_new.to(dtype=a_dtype), b_new.to(dtype=b_dtype))


def _active_lora_parameter_pairs(model):
    return [
        (a_name, a_param, b_param)
        for a_name, a_param, _, b_param in _lora_parameter_pairs(model)
        if a_param.grad is not None and b_param.grad is not None
    ]


def _state_for_layer(state, key, shape, device):
    layer_state = state.get(key)
    if layer_state is None or tuple(layer_state["m"].shape) != tuple(shape):
        layer_state = {
            "step": 0,
            "m": torch.zeros(shape, device=device, dtype=torch.float32),
            "v": torch.zeros(shape, device=device, dtype=torch.float32),
        }
        state[key] = layer_state
    else:
        layer_state["m"] = layer_state["m"].to(device=device, dtype=torch.float32)
        layer_state["v"] = layer_state["v"].to(device=device, dtype=torch.float32)
    return layer_state


def apply_barylora_local_update(model, cfg, state, lr, round_idx=None):
    """Update persistent moments, apply the zero prior, then retract to rank r."""
    if round_idx is not None and round_idx < _cfg_value(cfg, "bary_warmup_rounds", 0):
        return
    beta1 = _cfg_value(cfg, "bary_moment_beta1", 0.9)
    beta2 = _cfg_value(cfg, "bary_moment_beta2", 0.999)
    eps = _cfg_value(cfg, "bary_eps", 1e-8)
    lr_scale = _cfg_value(cfg, "bary_step_scale", 1.0)
    prior_scale = _cfg_value(cfg, "bary_prior_scale", 0.0)
    prior_warmup = _cfg_value(cfg, "bary_prior_warmup_steps", 1)
    client_weight = _cfg_value(cfg, "bary_client_prior_weight", 1.0)
    for a_name, a_param, b_param in _active_lora_parameter_pairs(model):
        grad_mu = _recover_effective_update_direction(
            a_param,
            b_param,
            a_param.grad,
            b_param.grad,
            _cfg_value(cfg, "bary_damping", 1e-4),
            damping_mode=_cfg_value(cfg, "bary_damping_mode", "relative"),
            damping_rel=_cfg_value(cfg, "bary_damping_rel", 1e-7),
            projection=_cfg_value(cfg, "bary_direction_projection", "orthogonal"),
        )
        with torch.no_grad():
            mu = b_param.detach().to(dtype=torch.float32) @ a_param.detach().to(
                dtype=torch.float32
            )
            layer = _state_for_layer(state, a_name, mu.shape, mu.device)
            layer["step"] += 1
            step = layer["step"]
            g_like = -grad_mu
            layer["m"].mul_(beta1).add_(g_like, alpha=1.0 - beta1)
            layer["v"].mul_(beta2).addcmul_(g_like, g_like, value=1.0 - beta2)
            m_hat = layer["m"] / (1.0 - beta1**step)
            v_hat = layer["v"] / (1.0 - beta2**step)
            activity = activity_from_second_moment(
                v_hat,
                floor=0.0,
                cap=_cfg_value(cfg, "bary_activity_cap", 10.0),
                var_scale=_cfg_value(cfg, "bary_var_scale", 1.0),
                var_eps=_cfg_value(cfg, "bary_var_eps", 1e-8),
                gamma=1.0,
            )
            ramp = (
                1.0
                if prior_warmup <= 0
                else min(1.0, float(step) / float(prior_warmup))
            )
            prior_pull = -prior_scale * float(client_weight) * ramp * activity * mu
            update = lr * lr_scale * (m_hat + prior_pull) / (torch.sqrt(v_hat) + eps)
            a_new, b_new = _rank_project_on_device(
                mu + update,
                a_param.shape[0],
                a_param.dtype,
                b_param.dtype,
                mode=_cfg_value(cfg, "bary_local_rank_return", "exact"),
                randomized_q=_cfg_value(cfg, "bary_local_rank_return_q", 0),
                randomized_niter=_cfg_value(cfg, "bary_local_rank_return_niter", 2),
            )
            a_param.copy_(a_new)
            b_param.copy_(b_new)
        a_param.grad = None
        b_param.grad = None
