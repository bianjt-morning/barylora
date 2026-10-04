"""Task trainers retained by the paper release."""

import importlib


def get_trainer(
    model=None,
    data=None,
    device=None,
    config=None,
    only_for_eval=False,
    is_attacker=False,
    monitor=None,
):
    paths = {
        "gluetrainer": ("federatedscope.glue.trainer.trainer", "GLUETrainer"),
        "llmtrainer": ("federatedscope.llm.trainer.trainer", "LLMTrainer"),
        "general": ("federatedscope.core.trainers", "GeneralTorchTrainer"),
    }
    name = config.trainer.type.lower()
    if name == "none":
        return None
    if name not in paths:
        raise ValueError(f"Trainer {name} is not in the paper release")
    module, class_name = paths[name]
    cls = getattr(importlib.import_module(module), class_name)
    return cls(
        model=model,
        data=data,
        device=device,
        config=config,
        only_for_eval=only_for_eval,
        monitor=monitor,
    )
