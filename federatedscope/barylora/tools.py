"""LoRA wire format, activity summaries and exact QR-core rank return."""

import copy
import hashlib
from collections import OrderedDict
import torch

BARYLORA_METADATA_PREFIX = "__barylora__:"
WRAPPER_NAME_PREFIX = "model."


def is_lora_a_key(key):
    return "lora_A" in key


def is_lora_b_key(key):
    return "lora_B" in key


def _to_tensor(value):
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def bare_param_name(key):
    """Match AdapterModel named parameters to its unprefixed wire keys."""
    if isinstance(key, str) and key.startswith(WRAPPER_NAME_PREFIX):
        return key[len(WRAPPER_NAME_PREFIX) :]
    return key


def expand_trainable_names(trainable_para_names):
    names = set(trainable_para_names or ())
    if not names:
        return names
    bare = {bare_param_name(name) for name in names}
    return names | bare | {WRAPPER_NAME_PREFIX + name for name in bare}


def barylora_activity_key(a_key):
    digest = hashlib.sha1(a_key.encode("utf-8")).hexdigest()[:16]
    return f"{BARYLORA_METADATA_PREFIX}activity:{digest}"


def is_barylora_metadata_key(key):
    return isinstance(key, str) and key.startswith(BARYLORA_METADATA_PREFIX)


def split_lora_ab_keys(state):
    pairs = []
    a_keys = {key for key in state if is_lora_a_key(key)}
    for b_key in sorted((key for key in state if is_lora_b_key(key))):
        a_key = b_key.replace("lora_B", "lora_A")
        if a_key not in state:
            raise ValueError(f"Missing matching LoRA A key for {b_key}: {a_key}")
    for a_key in sorted(a_keys):
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in state:
            raise ValueError(f"Missing matching LoRA B key for {a_key}: {b_key}")
        a = _to_tensor(state[a_key])
        b = _to_tensor(state[b_key])
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"LoRA parameters must be 2D, got {a_key}{tuple(a.shape)} and {b_key}{tuple(b.shape)}"
            )
        if b.shape[1] != a.shape[0]:
            raise ValueError(
                f"LoRA rank mismatch for {a_key}/{b_key}: A rank {a.shape[0]}, B rank {b.shape[1]}"
            )
        pairs.append((a_key, b_key))
    return pairs


def strip_lora_keys(state):
    return OrderedDict(
        (
            (key, _clone_value(value))
            for key, value in state.items()
            if not is_lora_a_key(key)
            and (not is_lora_b_key(key))
            and (not is_barylora_metadata_key(key))
        )
    )


def strip_lora_keys_from_client_feedback(client_feedback):
    return [
        (sample_size, strip_lora_keys(state)) for sample_size, state in client_feedback
    ]


def filter_barylora_upload_state(state, trainable_para_names):
    trainable = expand_trainable_names(trainable_para_names)
    result = OrderedDict()
    for key, value in state.items():
        is_lora_key = is_lora_a_key(key) or is_lora_b_key(key)
        is_trainable_non_lora = not is_lora_key and bare_param_name(key) in trainable
        if is_barylora_metadata_key(key) or is_lora_key or is_trainable_non_lora:
            result[key] = _clone_value(value)
    return result


def _apply_deterministic_sign(a, b):
    for idx in range(b.shape[1]):
        column = b[:, idx]
        if column.numel() == 0:
            continue
        pivot = torch.argmax(torch.abs(column))
        if column[pivot] < 0:
            b[:, idx] = -b[:, idx]
            a[idx, :] = -a[idx, :]
    return (a, b)


def _pad_rank_return(a, b, rank):
    if a.shape[0] == rank:
        return (a, b)
    pad_rank = rank - a.shape[0]
    if pad_rank < 0:
        return (a[:rank, :], b[:, :rank])
    a_pad = torch.zeros(pad_rank, a.shape[1], dtype=a.dtype, device=a.device)
    b_pad = torch.zeros(b.shape[0], pad_rank, dtype=b.dtype, device=b.device)
    return (torch.cat([a, a_pad], dim=0), torch.cat([b, b_pad], dim=1))


def qr_core_rank_return(weighted_bs, weighted_as, rank, out_dtype):
    """Project the weighted product using stacked factors and a small QR core."""
    if not weighted_bs:
        raise ValueError("qr_core_rank_return requires at least one client")
    left = torch.cat(
        [b.detach().to(device="cpu", dtype=torch.float32) for b in weighted_bs], dim=1
    )
    right = torch.cat(
        [a.detach().to(device="cpu", dtype=torch.float32) for a in weighted_as], dim=0
    )
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, s, vh_core = torch.linalg.svd(core, full_matrices=False)
    keep = min(rank, s.numel())
    sqrt_s = torch.sqrt(torch.clamp(s[:keep], min=0.0))
    b_new = q_left @ u_core[:, :keep] * sqrt_s.unsqueeze(0)
    a_new = sqrt_s.unsqueeze(1) * (vh_core[:keep, :] @ q_right.T)
    a_new, b_new = _pad_rank_return(a_new, b_new, rank)
    a_new, b_new = _apply_deterministic_sign(a_new, b_new)
    return (a_new.to(dtype=out_dtype), b_new.to(dtype=out_dtype))


def _get_cfg_value(cfg, path, default=None):
    cur = cfg
    for name in path.split("."):
        if not hasattr(cur, name):
            return default
        cur = getattr(cur, name)
    return cur


def _positive_cfg_float(cfg, path, default):
    value = float(_get_cfg_value(cfg, path, default))
    return value if value > 0.0 else float(default)


def _non_negative_cfg_float(cfg, path, default):
    value = float(_get_cfg_value(cfg, path, default))
    return value if value >= 0.0 else float(default)


def _unit_interval_cfg_float(cfg, path, default):
    value = float(_get_cfg_value(cfg, path, default))
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _activity_scalar_from_state(layer_state, cfg):
    if not isinstance(layer_state, dict) or "v" not in layer_state:
        return None
    v = _to_tensor(layer_state["v"]).detach().to(device="cpu", dtype=torch.float32)
    if v.numel() == 0:
        return None
    step = int(layer_state.get("step", 0) or 0)
    beta2 = _non_negative_cfg_float(cfg, "lora.bary_moment_beta2", 0.999)
    beta2 = min(beta2, 1.0 - 1e-12)
    if step > 0:
        bias_correction = 1.0 - beta2**step
        v_hat = v / bias_correction if bias_correction > 0.0 else v
    else:
        v_hat = v
    var_scale = _positive_cfg_float(cfg, "lora.bary_var_scale", 1.0)
    var_eps = _positive_cfg_float(cfg, "lora.bary_var_eps", 1e-08)
    activity_cap = _positive_cfg_float(cfg, "lora.bary_activity_cap", 10.0)
    floor = _positive_cfg_float(cfg, "lora.bary_kl_activity_floor", 1e-08)
    gamma = _non_negative_cfg_float(cfg, "lora.bary_kl_activity_gamma", 0.6)
    activity_cap = max(activity_cap, floor)
    mode = "variance"
    scalar = activity_from_second_moment(
        v_hat,
        mode=mode,
        floor=floor,
        cap=activity_cap,
        var_scale=var_scale,
        var_eps=var_eps,
        gamma=gamma,
        reduce_scalar=True,
    )
    if not torch.isfinite(scalar) or float(scalar.item()) <= 0.0:
        return None
    return scalar.detach().to(device="cpu", dtype=torch.float32).reshape(())


def _activity_value_from_state(state, a_key, cfg):
    key = barylora_activity_key(a_key)
    default = _positive_cfg_float(cfg, "lora.bary_kl_activity_default", 1.0)
    floor = _positive_cfg_float(cfg, "lora.bary_kl_activity_floor", 1e-08)
    if key not in state:
        return (max(default, floor), True)
    value = _to_tensor(state[key]).detach().to(device="cpu", dtype=torch.float32)
    if value.numel() == 0:
        return (max(default, floor), True)
    scalar = value.mean()
    if not torch.isfinite(scalar) or float(scalar.item()) <= 0.0:
        return (max(default, floor), True)
    return (max(float(scalar.item()), floor), False)


def _effective_activity_reliability(cfg, missing_count, total_count):
    reliability = _unit_interval_cfg_float(
        cfg, "lora.bary_kl_activity_reliability", 1.0
    )
    if total_count <= 0:
        return 0.0
    complete_fraction = max(0.0, min(1.0, (total_count - missing_count) / total_count))
    return reliability * complete_fraction


def _aggregation_weights(client_feedback, ignore_weight):
    if not client_feedback:
        raise ValueError("No client feedback for NG-VKLC aggregation")
    if ignore_weight:
        weight = 1.0 / len(client_feedback)
        return [weight for _ in client_feedback]
    total = sum((sample_size for sample_size, _ in client_feedback))
    if total <= 0:
        raise ValueError("Total sample size must be positive")
    return [sample_size / total for sample_size, _ in client_feedback]


def _validate_rank(cfg, rank, a_key):
    configured = _get_cfg_value(cfg, "lora.bary_rank", -1)
    if configured > 0 and configured != rank:
        raise ValueError(
            f"Configured lora.bary_rank={configured} does not match {a_key} rank={rank}"
        )


def activity_from_second_moment(
    v_hat,
    mode="variance",
    floor=0.0,
    cap=10.0,
    var_scale=1.0,
    var_eps=1e-8,
    gamma=1.0,
    reduce_scalar=False,
):
    """Paper activity: bounded variance; upload the powered layer mean."""
    if mode != "variance":
        raise ValueError("The paper release supports variance activity only")
    floor, cap = float(floor), max(float(cap), float(floor))
    variance = _to_tensor(v_hat).detach().to(dtype=torch.float32)
    variance = torch.clamp(
        (variance + float(var_eps)) / float(var_scale), min=floor, max=cap
    )
    activity = torch.nan_to_num(variance, nan=cap, posinf=cap, neginf=floor)
    if not reduce_scalar:
        return activity
    scalar = torch.clamp(activity.mean(), min=floor)
    if float(gamma) != 1.0:
        scalar = scalar.pow(float(gamma))
    return scalar.detach().to(dtype=torch.float32).reshape(())


def attach_barylora_state_summary(state, bary_state, cfg):
    """Attach one activity scalar per layer when state masses are enabled."""
    result = OrderedDict(
        (key, _clone_value(value))
        for key, value in state.items()
        if not is_barylora_metadata_key(key)
    )
    enabled = _get_cfg_value(
        cfg, "lora.bary_mass_rule", "off"
    ) == "state" and _get_cfg_value(cfg, "lora.bary_send_state_summary", True)
    if not enabled:
        return result
    bary_state = bary_state or {}
    for a_key, _ in split_lora_ab_keys(state):
        layer_state = bary_state.get(a_key)
        if layer_state is None:
            matches = [
                value
                for key, value in bary_state.items()
                if isinstance(key, str) and key.endswith(a_key)
            ]
            if len(matches) == 1:
                layer_state = matches[0]
        scalar = _activity_scalar_from_state(layer_state, cfg)
        if scalar is not None:
            result[barylora_activity_key(a_key)] = scalar
    return result
