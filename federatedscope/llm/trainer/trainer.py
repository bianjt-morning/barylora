import json
import os

import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken   # added by me, for gsm8k evaluation
from federatedscope.llm.misc.fschat import FSChatBot_My   # added by me, for gsm8k evaluation
from federatedscope.llm.eval.eval_for_gsm8k.eval import *   # added by me, for gsm8k evaluation
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT   # added by me, for gsm8k evaluation
from federatedscope.barylora.local_update import apply_barylora_local_update

logger = logging.getLogger(__name__)


def _clear_lora_grads(model):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.grad = None


def _ng_warmup_active(cfg, round_idx):
    """Return True while the current round is still inside NG warmup.

    Warmup means the LoRA factors are trained by the ordinary optimizer
    (AdamW), exactly as in a plain LoRA run; the natural-gradient update only
    takes over from round ``lora.bary_warmup_rounds`` onward.  This raises
    instead of guessing when warmup is configured but the round index is
    unavailable, because the two training paths are mutually exclusive and
    silently picking one would make the arm a no-op or skip its warmup.
    """
    warmup_rounds = getattr(cfg.lora, "bary_warmup_rounds", 0)
    try:
        warmup_rounds = int(warmup_rounds)
    except (TypeError, ValueError):
        raise RuntimeError(
            "lora.bary_warmup_rounds must be an integer, got "
            f"{warmup_rounds!r}.")
    if warmup_rounds <= 0:
        return False
    if round_idx is None:
        raise RuntimeError(
            "NG-VKLC warmup is enabled (lora.bary_warmup_rounds="
            f"{warmup_rounds}) but the current round could not be determined "
            "(getattr(ctx, 'cur_round', getattr(ctx, 'state', None)) is "
            "None). Refusing to guess between warmup and natural-gradient "
            "training; ensure ctx.cur_round is set.")
    return round_idx < warmup_rounds


# Generative-eval registry.
#
# Before 2026-09-13 the gate below was hardcoded to `dataset_name ==
# 'gsm8k'`, so setting `eval.llm_generation: True` on any other dataset was
# a silent no-op: no `{split}_acc` key appeared and nothing warned.  A
# silently-absent metric is worse than an absent feature, because it reads
# as "we measured it and it was fine".  The gate is now driven by this
# registry and emits a warning when a run asks for generative eval on a
# dataset with no defined scorer.
#
# Adding a dataset here REQUIRES a defined correctness criterion for its
# outputs.  Do not add one just to make a table column appear.
GENERATION_EVAL_DATASETS = ('gsm8k', )


def _generation_eval_dataset(ctx):
    """Return the canonical dataset name if a scorer exists for it."""
    dataset_name = ctx.cfg.data.type.split('@')[0].lower()
    if dataset_name in GENERATION_EVAL_DATASETS:
        return dataset_name
    return None


def _should_run_generation_eval(ctx):
    if ctx.cur_mode not in (MODE.VAL, MODE.TEST):
        return False
    if not hasattr(ctx, 'val_loader_copy'):
        return False
    if not getattr(ctx.cfg.eval, 'llm_generation', False):
        return False
    if _generation_eval_dataset(ctx) is not None:
        return True
    logger.warning(
        "eval.llm_generation=True but dataset '%s' has no registered "
        "generative scorer (registered: %s); skipping generative eval so "
        "that no undefined accuracy is reported.",
        ctx.cfg.data.type.split('@')[0],
        ', '.join(GENERATION_EVAL_DATASETS),
    )
    return False


# Backwards-compatible alias: the old name was GSM8K-specific and is kept
# only so existing imports keep working.
_should_run_gsm8k_generation_eval = _should_run_generation_eval


def _get_gsm8k_generation_max_samples(ctx):
    max_samples = getattr(ctx.cfg.eval, 'llm_generation_max_samples', -1)
    if max_samples is None:
        return -1
    return int(max_samples)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y')
    return bool(value)


def _build_gsm8k_generation_kwargs(ctx):
    eval_cfg = ctx.cfg.eval
    max_new_tokens = getattr(eval_cfg, 'llm_generation_max_new_tokens', 256)
    if max_new_tokens is None:
        max_new_tokens = 256

    do_sample = _as_bool(
        getattr(eval_cfg, 'llm_generation_do_sample', False))
    generate_kwargs = {
        'max_new_tokens': int(max_new_tokens),
        'do_sample': do_sample,
    }
    if do_sample:
        generate_kwargs['temperature'] = float(
            getattr(eval_cfg, 'llm_generation_temperature', 0.8))
        generate_kwargs['top_p'] = float(
            getattr(eval_cfg, 'llm_generation_top_p', 0.95))
    return generate_kwargs


def _get_gsm8k_prediction_log_path(ctx):
    explicit_path = getattr(
        ctx.cfg.eval,
        'llm_generation_prediction_path',
        '',
    )
    if explicit_path:
        return explicit_path

    outdir = getattr(ctx.cfg, 'outdir', '') or os.getcwd()
    split = getattr(ctx, 'cur_split', 'unknown')
    round_idx = getattr(ctx, 'cur_round', getattr(ctx, 'state', 'unknown'))
    filename = f'{split}_round{round_idx}_pid{os.getpid()}.jsonl'
    return os.path.join(outdir, 'gsm8k_generation_predictions', filename)


def _write_gsm8k_prediction_record(path, record):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + '\n')


class LLMTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        
        # added by me, for gsm8k evaluation
        if not hasattr(ctx, 'val_loader_copy') and hasattr(ctx, 'val_loader'):
            ctx.val_loader_copy = ctx.val_loader
        
    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the activity or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            if ctx.cfg.lora.method == "barylora" and \
                    ctx.cfg.lora.bary_local_update:
                round_idx = getattr(
                    ctx, "cur_round", getattr(ctx, "state", None))
                # During warmup the natural-gradient step is skipped so that
                # the ordinary optimizer trains lora_A/lora_B exactly as in a
                # plain LoRA run; the NG update replaces AdamW for the LoRA
                # factors only from round bary_warmup_rounds onward.  The NG
                # step and _clear_lora_grads must be skipped together:
                # clearing the grads WITHOUT applying an NG update makes
                # AdamW skip the LoRA factors, so B never leaves its zero
                # initialisation and the recovered product gradient stays
                # identically zero forever.
                if not _ng_warmup_active(ctx.cfg, round_idx):
                    if not hasattr(ctx, "barylora_state"):
                        ctx.barylora_state = {}
                    lr = ctx.optimizer.param_groups[0].get("lr")
                    if lr is None:
                        lr = ctx.cfg.train.optimizer.lr
                    apply_barylora_local_update(
                        model=ctx.model,
                        cfg=ctx.cfg,
                        state=ctx.barylora_state,
                        lr=lr,
                        round_idx=round_idx,
                    )
                    _clear_lora_grads(ctx.model)

            ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        
        # added by me, evaluating on GSM8K dataset
        if _should_run_generation_eval(ctx):
            fschatbot = FSChatBot_My(ctx.model.cpu(), ctx.cfg)
            answers = []
            max_samples = _get_gsm8k_generation_max_samples(ctx)
            prediction_log_path = None
            if _as_bool(getattr(ctx.cfg.eval,
                                'llm_generation_save_predictions',
                                True)):
                prediction_log_path = _get_gsm8k_prediction_log_path(ctx)
            for batch in ctx.val_loader_copy:
                for instruction, _, output in zip(batch['instruction'], batch['input'], batch['output']):
                    if max_samples >= 0 and len(answers) >= max_samples:
                        break
                    input_text = build_prompt(instruction, N_SHOT, COT_FLAG)
                    generate_kwargs = _build_gsm8k_generation_kwargs(ctx)
                    raw_completion = fschatbot.generate(input_text,
                                                        generate_kwargs)
                    model_completion = truncate_gsm8k_completion(
                        raw_completion)
                    model_answer = clean_answer(model_completion)
                    is_cor = is_correct(model_answer, output)
                    answers.append(is_cor)
                    if prediction_log_path is not None:
                        _write_gsm8k_prediction_record(
                            prediction_log_path,
                            {
                                'index': len(answers) - 1,
                                'split': ctx.cur_split,
                                'round': getattr(ctx, 'cur_round', None),
                                'question': instruction,
                                'gold': extract_answer_from_output(output),
                                'model_answer': model_answer,
                                'correct': bool(is_cor),
                                'completion': model_completion,
                                'raw_completion': raw_completion,
                            },
                        )
                    print(f'Question: {instruction}\n\n'
                          f'Answers: {extract_answer_from_output(output)}\n\n'
                          f'Model Answers: {model_answer}\n\n'
                          f'Model Completion: {model_completion}\n\n'
                          f'Is correct: {is_cor}\n\n')

                    print(f'Num of total question: {len(answers)}, '
                          f'correct num: {sum(answers)}, '
                          f'correct rate: {float(sum(answers))/len(answers)}.')
                if max_samples >= 0 and len(answers) >= max_samples:
                    break
            if len(answers) > 0:
                eval_results[f'{ctx.cur_split}_acc'] = \
                    float(sum(answers)) / len(answers)
        
        setattr(ctx, 'eval_metrics', eval_results)
                
        # # TODO: make this as a hook function
        # # Move trainable part to `cpu`, which can save memory but cost time
        # if ctx.cfg.llm.adapter.mv_to_cpu:
        #     for p in ctx.model.parameters():
        #         if p.requires_grad:
        #             p.data = p.to('cpu')
        #             if p.grad is not None:
        #                 p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
