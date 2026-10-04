"""Bounded state masses and weighted product-space consensus."""

import math
from collections import OrderedDict
import torch
from federatedscope.barylora.tools import (
    _get_cfg_value,
    _positive_cfg_float,
    _aggregation_weights,
    _effective_activity_reliability,
    _activity_value_from_state,
    _to_tensor,
    _validate_rank,
    split_lora_ab_keys,
    qr_core_rank_return,
    is_lora_a_key,
    is_lora_b_key,
    is_barylora_metadata_key,
)


def layer_activities(states, a_key, cfg):
    activities = []
    missing_count = 0
    for state in states:
        value, used_default = _activity_value_from_state(state, a_key, cfg)
        activities.append(float(value))
        if used_default:
            missing_count += 1
    return (activities, missing_count)


def head_inheritance_keys(state):
    keys = []
    for key, value in state.items():
        if is_lora_a_key(key) or is_lora_b_key(key):
            continue
        if is_barylora_metadata_key(key):
            continue
        if not _to_tensor(value).is_floating_point():
            continue
        keys.append(key)
    return keys


def weighted_state_average(states, keys, weights, label):
    result = OrderedDict()
    for key in keys:
        first = _to_tensor(states[0][key])
        accumulator = None
        for weight, state in zip(weights, states):
            if key not in state:
                raise ValueError("{} missing non-LoRA key {}".format(label, key))
            value = (
                _to_tensor(state[key]).detach().to(device="cpu", dtype=torch.float32)
            )
            if tuple(value.shape) != tuple(first.shape):
                raise ValueError(
                    "{} shape mismatch for {}: expected {}, got {}".format(
                        label, key, tuple(first.shape), tuple(value.shape)
                    )
                )
            term = value * float(weight)
            accumulator = term if accumulator is None else accumulator + term
        result[key] = accumulator.to(dtype=first.dtype).detach().clone()
    return result


def state_conditioned_masses(activities, base_weights, cfg, missing_count=0):
    """Eq. 11: normalize bounded activity tilts of the sample-size prior."""
    base = [float(weight) for weight in base_weights]
    count = len(base)
    if not count:
        return base
    usable = [
        float(value)
        for value in activities
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if len(usable) != count:
        return base
    mean = sum(usable) / count
    if not math.isfinite(mean) or mean <= 0.0:
        return base
    reliability = _effective_activity_reliability(cfg, missing_count, count)
    if reliability <= 0.0:
        return base
    bound = max(1.0, float(_positive_cfg_float(cfg, "lora.bary_mass_clip", 2.0)))
    factors = [min(max(value / mean, 1.0 / bound), bound) for value in usable]
    blended = [(1.0 - reliability) + reliability * factor for factor in factors]
    raw = [max(weight, 0.0) * factor for weight, factor in zip(base, blended)]
    total = sum(raw)
    return [value / total for value in raw] if total > 0.0 else base


def aggregate_barylora_consensus(client_feedback, cfg):
    """Return LoRA factors, plus the optional inherited task head.

    'off' retains the shipped LLM sample-mass protocol; 'state' is the
    final GLUE state-conditioned protocol. Task metrics belong to trainers.
    """
    mode = _get_cfg_value(cfg, "lora.bary_mass_rule", "off")
    if mode not in ("off", "state"):
        raise ValueError("bary_mass_rule must be off or state")
    base = _aggregation_weights(
        client_feedback, _get_cfg_value(cfg, "federate.ignore_weight", False)
    )
    states = [state for _, state in client_feedback]
    pairs = split_lora_ab_keys(states[0])
    result = OrderedDict()
    sums = [0.0] * len(states)
    for a_key, b_key in pairs:
        first_a, first_b = _to_tensor(states[0][a_key]), _to_tensor(states[0][b_key])
        rank = first_a.shape[0]
        _validate_rank(cfg, rank, a_key)
        if mode == "state":
            activities, missing = layer_activities(states, a_key, cfg)
            weights = state_conditioned_masses(activities, base, cfg, missing)
        else:
            weights = base
        weighted_as, weighted_bs = [], []
        for index, (weight, state) in enumerate(zip(weights, states)):
            if a_key not in state or b_key not in state:
                raise ValueError(f"Missing LoRA pair {a_key}/{b_key}")
            a = _to_tensor(state[a_key]).detach().to(device="cpu", dtype=torch.float32)
            b = _to_tensor(state[b_key]).detach().to(device="cpu", dtype=torch.float32)
            if a.shape != first_a.shape or b.shape != first_b.shape:
                raise ValueError(f"LoRA shape mismatch for {a_key}/{b_key}")
            scale = math.sqrt(max(weight, 0.0))
            weighted_as.append(a * scale)
            weighted_bs.append(b * scale)
            sums[index] += weight
        a_new, b_new = qr_core_rank_return(
            weighted_bs, weighted_as, rank, first_a.dtype
        )
        result[a_key], result[b_key] = a_new.detach().clone(), b_new.detach().clone()
    if (
        pairs
        and mode == "state"
        and _get_cfg_value(cfg, "lora.bary_head_inheritance", "off") == "inherit"
    ):
        totals = [value / float(len(pairs)) for value in sums]
        total = sum(totals)
        weights = (
            [value / total for value in totals]
            if total > 0.0
            else [1.0 / len(states)] * len(states)
        )
        result.update(
            weighted_state_average(
                states, head_inheritance_keys(states[0]), weights, "head pooling"
            )
        )
    return result
