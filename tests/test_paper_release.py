"""Mathematical and publication-boundary checks; no training downloads."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import pytest

from federatedscope.barylora.consensus import state_conditioned_masses
from federatedscope.barylora.local_update import _recover_effective_update_direction
from federatedscope.barylora.tools import (
    qr_core_rank_return,
    filter_barylora_upload_state,
    barylora_activity_key,
    _activity_value_from_state,
)


def test_training_tree_has_no_diagnostic_or_historical_imports():
    root = Path(__file__).resolve().parents[1] / "federatedscope"
    forbidden = (
        "federatedscope.core.diagnostics",
        "federatedscope.barylora.moments",
        "federatedscope.barylora.fallback",
        "federatedscope.llm.model.ravan_adapter",
    )
    found = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(forbidden):
                    found.append(f"{path.name}:{node.lineno}:{node.module}")
    assert not found, found


def test_qr_core_preserves_product_under_independent_coordinates():
    torch.manual_seed(42)
    bs = [torch.randn(17, 4) for _ in range(3)]
    ass = [torch.randn(4, 13) for _ in range(3)]
    a, b = qr_core_rank_return(bs, ass, 4, torch.float32)
    qs = [torch.linalg.qr(torch.randn(4, 4))[0] for _ in range(3)]
    rotated_a, rotated_b = qr_core_rank_return(
        [bb @ q for bb, q in zip(bs, qs)],
        [q.T @ aa for q, aa in zip(qs, ass)],
        4,
        torch.float32,
    )
    torch.testing.assert_close(rotated_b @ rotated_a, b @ a, rtol=3e-5, atol=1e-5)


def test_direction_recovers_minimum_norm_compatible_gradient():
    torch.manual_seed(42)
    a, b, g = torch.randn(4, 13), torch.randn(17, 4), torch.randn(17, 13)
    got = _recover_effective_update_direction(a, b, b.T @ g, g @ a.T, 1e-4)
    pb, pa = b @ torch.linalg.pinv(b), torch.linalg.pinv(a) @ a
    expected = pb @ g + g @ pa - pb @ g @ pa
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=3e-6)


def test_mass_returns_sample_prior_when_all_summaries_missing():
    cfg = NS(lora=NS(bary_mass_clip=2.0, bary_kl_activity_reliability=1.0))
    got = state_conditioned_masses(
        [1.0, 1.0, 1.0], [0.2, 0.3, 0.5], cfg, missing_count=3
    )
    assert got == [0.2, 0.3, 0.5]


def test_wrapper_name_filter_keeps_head_but_not_backbone():
    state = {
        "layer.lora_A.weight": torch.ones(2, 3),
        "layer.lora_B.weight": torch.ones(4, 2),
        "classifier.weight": torch.ones(3, 4),
        "embedding.weight": torch.ones(5, 4),
    }
    got = filter_barylora_upload_state(state, ["model.classifier.weight"])
    assert set(got) == set(state) - {"embedding.weight"}
    assert got["classifier.weight"].data_ptr() != state["classifier.weight"].data_ptr()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_summary_uses_sample_fallback(value):
    cfg = NS(lora=NS(bary_kl_activity_default=1.0, bary_kl_activity_floor=1e-8))
    scalar, missing = _activity_value_from_state(
        {barylora_activity_key("layer.lora_A.weight"): torch.tensor(value)},
        "layer.lora_A.weight",
        cfg,
    )
    assert missing and scalar == 1.0


@pytest.mark.parametrize("mode", ["state", "off"])
def test_shipped_configs_parse_without_historical_switches(mode):
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.configs.cfg_llm import assert_llm_cfg

    root = Path(__file__).resolve().parents[1]
    files = (
        [root / "configs/barylora_glue.yaml"]
        if mode == "state"
        else list((root / "configs").glob("barylora_gsm8k_*.yaml"))
    )
    for path in files:
        cfg = global_cfg.clone()
        cfg.merge_from_file(str(path))
        assert_llm_cfg(cfg)
        assert cfg.lora.bary_mass_rule == mode
        assert "bary_log_metrics" not in cfg.lora


def test_historical_algorithm_and_mass_modes_are_rejected():
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.configs.cfg_llm import assert_llm_cfg

    for key, value in [
        ("method", "flora"),
        ("bary_mass_rule", "shadow"),
        ("bary_mass_rule", "loop"),
        ("bary_direction_projection", "half_sum"),
    ]:
        cfg = global_cfg.clone()
        setattr(cfg.lora, key, value)
        with pytest.raises(ValueError):
            assert_llm_cfg(cfg)
