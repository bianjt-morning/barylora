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
from federatedscope.glue.model.adapter_builder import AdapterModel
from federatedscope.glue.dataloader.dataloader import canonical_glue_task_name
from federatedscope.barylora.local_update import apply_barylora_local_update
from evaluate import load
# from datasets import load_metric
import numpy as np
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ``evaluate.load('glue', task, trust_remote_code=True)`` is called once per
# client per round (50 clients x 250 rounds = 12500 times per arm).  It is not
# covered by HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE --
# ``evaluate`` is a third library with its own switch -- so each call issued
# HTTP HEAD requests to the hub (1.44-1.75 s measured).  The returned metric
# object is stateless for ``compute`` and safe to reuse, so cache it per task
# name: at most one ``load`` per distinct task instead of one per call.
# ``HF_EVALUATE_OFFLINE=1`` is set by the GLUE protocol launcher so the
# remaining first call is also offline.
_GLUE_METRIC_CACHE: dict = {}


def _get_glue_metric(glue_task_name):
    metric = _GLUE_METRIC_CACHE.get(glue_task_name)
    if metric is None:
        metric = load('glue', glue_task_name, trust_remote_code=True)
        _GLUE_METRIC_CACHE[glue_task_name] = metric
    return metric


def _clear_lora_grads(model):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.grad = None


# Local prior correction: cache the training shard's class counts.
_LPC_CACHE = {}
_LPC_WARNED = set()


def _lpc_unwrap(obj):
    """Reach the training shard behind its DataLoader."""
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        inner = getattr(obj, 'dataset', None)
        if inner is None or inner is obj:
            break
        obj = inner
    return obj


def _lpc_list_labels(ds):
    """Labels of a per-client list/tuple of sample dicts (FedScope's GLUE
    path materialises exactly this behind the DataLoader)."""
    labels = []
    for item in ds:
        if isinstance(item, dict):
            for key in ('label', 'labels', 'y'):
                if key in item:
                    labels.append(int(item[key]))
                    break
            else:
                return None
        elif isinstance(item, (list, tuple)):
            labels.append(int(item[-1]))
        else:
            return None
    return labels


def _lpc_counts_from_dataset(ds, n_labels):
    """Per-class counts for whatever object sits behind the DataLoader."""
    labels = None
    try:
        labels = ds['label']                      # HF Dataset / column dict
    except Exception:
        labels = None
    if labels is None and isinstance(ds, (list, tuple)):
        labels = _lpc_list_labels(ds)
    if labels is None:
        return None
    idx = getattr(ds, 'indices', None)
    if idx is not None:
        labels = [labels[k] for k in idx]
    counts = torch.zeros(n_labels, dtype=torch.float)
    for y in labels:
        yi = int(y)
        if 0 <= yi < n_labels:
            counts[yi] += 1
    return counts


def _local_label_histogram(ctx):
    """Per-class sample counts of this client's TRAIN shard, or None."""
    data = getattr(ctx, 'data', None)
    ds = data.get('train', None) if isinstance(data, dict) else None
    if ds is None:
        return None

    n_labels = int(getattr(ctx.cfg.data, 'num_labels', 0) or 0)
    if n_labels <= 0:
        return None
    obj = _lpc_unwrap(ds)
    key = id(obj)
    if key in _LPC_CACHE:
        return _LPC_CACHE[key]

    counts = None
    try:
        counts = _lpc_counts_from_dataset(obj, n_labels)
    except Exception as exc:
        logger.warning('local_prior_correction: label scan failed (%s)', exc)
        counts = None

    if counts is None:
        if key not in _LPC_WARNED:
            _LPC_WARNED.add(key)
            logger.warning(
                'local_prior_correction: no label column; this client '
                'keeps the unweighted loss.')
        _LPC_CACHE[key] = None
        return None

    _LPC_CACHE[key] = counts
    return counts


def _local_class_weights(ctx):
    """Inverse-frequency class weights for this client, mean 1 over the
    classes the shard actually contains.  None = leave the loss alone."""
    counts = _local_label_histogram(ctx)
    if counts is None:
        return None
    present = counts > 0
    if int(present.sum()) <= 1:
        return None                       # single-class shard: nothing to do
    freq = counts / counts.sum().clamp(min=1.0)
    w = torch.zeros_like(counts)
    w[present] = 1.0 / freq[present].clamp(min=1e-3)
    w[present] = w[present] / w[present].mean()
    return w.to(ctx.device)


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


class GLUETrainer(GeneralTorchTrainer):
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
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)    # modified by me, for GLUE

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        
        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        preds = outputs.logits.argmax(dim=-1)  # modified by me, for GLUE
        loss = outputs.loss
        if getattr(ctx.cfg.lora, 'local_prior_correction', False) \
                and getattr(ctx, 'cur_split', None) == 'train' \
                and int(getattr(ctx.cfg.data, 'num_labels', 1) or 1) > 1:
            _w = _local_class_weights(ctx)
            if _w is not None:
                loss = F.cross_entropy(outputs.logits, labels, weight=_w)
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the activity or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(preds, LIFECYCLE.BATCH)

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
        
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        # cache label for evaluate, use extend not append
        ctx.ys_true.extend(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.extend(ctx.y_pred.detach().cpu().numpy())
        
    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss
        }
        # added by me, for GLUE
        glue_task_name = canonical_glue_task_name(ctx.cfg.data.type.split('@')[0])
        glue_metric = _get_glue_metric(glue_task_name)
        eval_metric = glue_metric.compute(predictions=ctx.ys_pred, references=ctx.ys_true)
        for k, v in eval_metric.items():
            eval_results[f'{ctx.cur_split}_{k}'] = v
        

        setattr(ctx, 'eval_metrics', eval_results)
        
        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

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


def call_glue_trainer(trainer_type):
    if trainer_type == 'gluetrainer':
        trainer_builder = GLUETrainer
        return trainer_builder


register_trainer('gluetrainer', call_glue_trainer)


# ---------------------------------------------------------------------------
