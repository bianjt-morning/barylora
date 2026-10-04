"""Configuration for the final LoRA paper paths (GLUE and causal LLM)."""

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_llm_cfg(cfg):
    cfg.llm = CN()
    cfg.llm.tok_len = 128
    cfg.llm.retry_on_nan_loss = False
    cfg.llm.cache = CN()
    cfg.llm.cache.model = ""
    cfg.llm.chat = CN()
    cfg.llm.chat.max_history_len = 10
    cfg.llm.chat.max_len = 100
    cfg.llm.deepspeed = CN()
    cfg.llm.deepspeed.use = False
    cfg.llm.deepspeed.ds_config = ""
    cfg.llm.adapter = CN()
    cfg.llm.adapter.use = False
    cfg.llm.adapter.args = [{}]
    cfg.llm.adapter.mv_to_cpu = False
    cfg.lora = CN()
    cfg.lora.method = ""
    cfg.lora.use_lora = True
    cfg.lora.bary_rank = -1
    cfg.lora.bary_rank_return = "qr_core"
    cfg.lora.bary_local_update = False
    cfg.lora.bary_aggregation_device = "cpu"
    cfg.lora.bary_moment_beta1 = 0.9
    cfg.lora.bary_moment_beta2 = 0.999
    cfg.lora.bary_eps = 1e-08
    cfg.lora.bary_step_scale = 1.0
    cfg.lora.bary_local_rank_return = "exact"
    cfg.lora.bary_local_rank_return_q = 0
    cfg.lora.bary_local_rank_return_niter = 2
    cfg.lora.bary_var_scale = 1.0
    cfg.lora.bary_var_eps = 1e-08
    cfg.lora.bary_activity_cap = 10.0
    cfg.lora.bary_prior_scale = 0.01
    cfg.lora.bary_prior_warmup_steps = 1
    cfg.lora.bary_client_prior_weight = 1.0
    cfg.lora.bary_damping = 0.0001
    cfg.lora.bary_damping_mode = "relative"
    cfg.lora.bary_damping_rel = 1e-07
    cfg.lora.bary_direction_projection = "orthogonal"
    cfg.lora.bary_warmup_rounds = 1
    cfg.lora.bary_kl_activity_gamma = 0.6
    cfg.lora.bary_kl_activity_floor = 1e-08
    cfg.lora.bary_kl_activity_default = 1.0
    cfg.lora.bary_kl_activity_reliability = 1.0
    cfg.lora.bary_mass_rule = "off"
    cfg.lora.bary_mass_clip = 2.0
    cfg.lora.bary_head_inheritance = "off"
    cfg.lora.local_prior_correction = False
    cfg.lora.bary_send_state_summary = True
    cfg.register_cfg_check_fun(assert_llm_cfg)


def assert_llm_cfg(cfg):
    lora = cfg.lora
    if lora.use_lora and lora.method not in ("", "barylora"):
        raise ValueError("This release supports lora.method=barylora only")
    options = {
        "bary_mass_rule": ("off", "state"),
        "bary_head_inheritance": ("off", "inherit"),
        "bary_rank_return": ("qr_core",),
        "bary_aggregation_device": ("cpu",),
        "bary_damping_mode": ("relative", "absolute"),
        "bary_direction_projection": ("orthogonal",),
        "bary_local_rank_return": ("exact", "randomized"),
    }
    for name, allowed in options.items():
        if getattr(lora, name) not in allowed:
            raise ValueError(f"lora.{name} must be one of {allowed}")
    for name in ("bary_moment_beta1", "bary_moment_beta2"):
        if not 0 <= getattr(lora, name) < 1:
            raise ValueError(f"lora.{name} must lie in [0, 1)")
    for name in (
        "bary_eps",
        "bary_var_scale",
        "bary_var_eps",
        "bary_activity_cap",
        "bary_kl_activity_floor",
        "bary_kl_activity_default",
    ):
        if getattr(lora, name) <= 0:
            raise ValueError(f"lora.{name} must be positive")
    for name in (
        "bary_warmup_rounds",
        "bary_local_rank_return_q",
        "bary_local_rank_return_niter",
    ):
        value = getattr(lora, name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"lora.{name} must be a nonnegative integer")
    if not 0 <= lora.bary_kl_activity_reliability <= 1 or lora.bary_mass_clip < 1:
        raise ValueError("Invalid bounded state-mass parameters")
    if lora.bary_local_update and cfg.llm.deepspeed.use:
        raise ValueError(
            "Product-space local updates require the standard PyTorch optimizer"
        )
    if lora.bary_local_update and cfg.federate.freeze_A:
        raise ValueError("Product-space updates require trainable A and B")
    if cfg.llm.adapter.use:
        args = cfg.llm.adapter.args[0]
        if (
            args.get("adapter_package", "peft") != "peft"
            or args.get("adapter_method", "lora").lower() != "lora"
        ):
            raise ValueError("The paper release supports PEFT LoRA only")


register_config("llm", extend_llm_cfg)
