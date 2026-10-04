"""Focused tests for the rank-r re-compression backend switch (ticket PERF_A1).

Run from the tree root with::

    PYTHONPATH=<tree_root> python -m pytest tests/test_barylora_rank_return.py -q

The exact path must be bit-identical to the historical inline formula; the
randomised path must not be materially worse than the rank-r truncation it is
approximating.  Nothing here trains a model or touches the GPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federatedscope.barylora.local_update import (  # noqa: E402
    RANK_RETURN_SVD_MODES,
    _rank_project_on_device,
)


def _hand_written_exact(mu, rank, a_dtype, b_dtype):
    """The pre-ticket body of ``_rank_project_on_device`` verbatim."""
    u, s, vh = torch.linalg.svd(
        mu.to(dtype=torch.float32), full_matrices=False
    )
    keep = min(rank, s.numel())
    sqrt_s = torch.sqrt(torch.clamp(s[:keep], min=0.0))
    b_new = u[:, :keep] * sqrt_s.unsqueeze(0)
    a_new = sqrt_s.unsqueeze(1) * vh[:keep, :]

    if keep < rank:
        a_pad = torch.zeros(
            rank - keep,
            a_new.shape[1],
            device=mu.device,
            dtype=a_new.dtype,
        )
        b_pad = torch.zeros(
            b_new.shape[0],
            rank - keep,
            device=mu.device,
            dtype=b_new.dtype,
        )
        a_new = torch.cat([a_new, a_pad], dim=0)
        b_new = torch.cat([b_new, b_pad], dim=1)

    return a_new.to(dtype=a_dtype), b_new.to(dtype=b_dtype)


SNR_SCALE = 0.17  # mu_norm ~ 0.17 measured on trained factors
NOISE_SCALE = 0.21  # ||upd|| / ||mu|| ~ 0.21 measured

_SHAPES = [(32, 24), (24, 32), (20, 16), (16, 20), (8, 8)]


def _make_mu(m, n, seed):
    gen = torch.Generator().manual_seed(seed)
    return SNR_SCALE * torch.randn(m, n, generator=gen)


def test_rank_return_svd_modes_constant():
    assert RANK_RETURN_SVD_MODES == ("exact", "randomized")


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("rank", [4, 8, 16])
def test_exact_is_bit_identical_to_historical_formula(shape, rank):
    mu = _make_mu(shape[0], shape[1], seed=1234 + rank)
    a_dtype, b_dtype = torch.float32, torch.float32
    a_old, b_old = _hand_written_exact(mu, rank, a_dtype, b_dtype)
    a_new, b_new = _rank_project_on_device(
        mu, rank, a_dtype, b_dtype, mode="exact", randomized_q=0,
        randomized_niter=2,
    )
    assert torch.equal(a_new, a_old), "exact a is not bit-identical"
    assert torch.equal(b_new, b_old), "exact b is not bit-identical"


@pytest.mark.parametrize("shape", [(6, 8), (8, 6), (3, 5)])
def test_exact_padding_branch_bit_identical(shape):
    # min(m, n) < rank exercises the zero-padding branch.
    rank = 10
    mu = _make_mu(shape[0], shape[1], seed=99)
    a_old, b_old = _hand_written_exact(
        mu, rank, torch.float32, torch.float32
    )
    a_new, b_new = _rank_project_on_device(
        mu, rank, torch.float32, torch.float32, mode="exact"
    )
    assert a_new.shape == a_old.shape == (rank, shape[1])
    assert b_new.shape == b_old.shape == (shape[0], rank)
    assert torch.equal(a_new, a_old)
    assert torch.equal(b_new, b_old)


def test_exact_default_kwargs_match_explicit_exact():
    mu = _make_mu(24, 32, seed=7)
    a_default, b_default = _rank_project_on_device(
        mu, 8, torch.float32, torch.float32
    )
    a_explicit, b_explicit = _rank_project_on_device(
        mu, 8, torch.float32, torch.float32, mode="exact", randomized_q=0,
        randomized_niter=2,
    )
    assert torch.equal(a_default, a_explicit)
    assert torch.equal(b_default, b_explicit)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("rank", [4, 8])
def test_randomized_error_within_1p5x_exact_truncation(shape, rank):
    # The rank must be strictly below min(m, n); at rank == min(m, n) the
    # exact path is exact and its residual is pure round-off, so a ratio
    # against it is meaningless (covered by the shape tests instead).
    if rank >= min(shape):
        pytest.skip("no genuine truncation error at rank == min(m, n)")
    mu = _make_mu(shape[0], shape[1], seed=4242 + rank)
    mu_norm = float(torch.linalg.norm(mu).item())

    a_ex, b_ex = _rank_project_on_device(
        mu, rank, torch.float32, torch.float32, mode="exact"
    )
    exact_err = float(torch.linalg.norm(mu - b_ex @ a_ex).item()) / mu_norm

    a_lr, b_lr = _rank_project_on_device(
        mu, rank, torch.float32, torch.float32, mode="randomized",
        randomized_q=rank + 16, randomized_niter=2,
    )
    randomized_err = float(torch.linalg.norm(mu - b_lr @ a_lr).item()) / mu_norm

    assert exact_err > 1e-4, (
        f"shape={shape} rank={rank}: exact truncation residual is at the "
        f"round-off floor ({exact_err:.3e}); test is not meaningful"
    )
    assert randomized_err <= 1.5 * exact_err, (
        f"shape={shape} rank={rank}: randomized_err={randomized_err:.3e} > "
        f"1.5 * exact_err={1.5 * exact_err:.3e}"
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_randomized_default_q_and_fp16_dtype_roundtrip(shape):
    rank = 8
    mu = _make_mu(shape[0], shape[1], seed=555)
    a_lr, b_lr = _rank_project_on_device(
        mu, rank, torch.float16, torch.float16, mode="randomized"
    )
    assert a_lr.shape == (rank, shape[1])
    assert b_lr.shape == (shape[0], rank)
    assert a_lr.dtype == torch.float16
    assert b_lr.dtype == torch.float16


def test_rank4_non_square_shape():
    mu = torch.randn(10, 4)
    a_new, b_new = _rank_project_on_device(
        mu, 4, torch.float32, torch.float32, mode="randomized"
    )
    assert a_new.shape == (4, 4)
    assert b_new.shape == (10, 4)
    # rank == min(10, 4): the target is reconstructible exactly, so the only
    # requirement is that the random range finder lands at the same (tiny)
    # round-off level as torch.linalg.svd, not a ratio of two round-offs.
    a_ex, b_ex = _rank_project_on_device(
        mu, 4, torch.float32, torch.float32, mode="exact"
    )
    randomized_abs = float(torch.linalg.norm(mu - b_new @ a_new).item())
    exact_abs = float(torch.linalg.norm(mu - b_ex @ a_ex).item())
    mu_norm = float(torch.linalg.norm(mu).item())
    assert randomized_abs <= max(1e-4 * mu_norm, 10.0 * exact_abs)


def test_invalid_mode_raises_value_error():
    mu = torch.randn(8, 8)
    with pytest.raises(ValueError) as excinfo:
        _rank_project_on_device(mu, 4, torch.float32, torch.float32,
                                      mode="bogus")
    message = str(excinfo.value)
    assert "bary_local_rank_return" in message
    assert "exact" in message and "randomized" in message
