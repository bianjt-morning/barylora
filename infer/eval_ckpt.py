#!/usr/bin/env python
"""Checkpoint-only GLUE inference for BaryLoRA checkpoints.

Given an already-trained checkpoint produced by the release/staging framework
(``federatedscope`` glue trainer), this script:

  1. rebuilds the exact ``FacebookAI/roberta-large`` +
     PEFT LoRA (r=4, alpha=8, dropout=0.05, target modules ``query``/``value``)
     + sequence-classification head, following
     ``federatedscope/glue/model/model_builder.py`` and
     ``federatedscope/glue/model/adapter_builder.py``;
  2. loads the checkpoint's trainable state dict (LoRA A/B factors + the
     ``classifier`` modules_to_save head);
  3. tokenizes the GLUE validation split exactly as
     ``federatedscope/glue/dataloader/dataloader.py``
     (``padding='max_length'``, ``max_length=tok_len``, ``truncation=True``);
  4. runs a single validation pass (no training, no optimizer) and writes a
     verifiable ``metrics.json``.

The script is self-contained (transformers + peft + datasets + evaluate) and
uses the same model and data preprocessing conventions as the training path.
"""

import argparse
import hashlib
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

# --- copied verbatim from federatedscope/glue/dataloader/dataloader.py ------
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
_TASK_ALIASES = {"mnli-m": "mnli", "mnli-mm": "mnli", "sst-2": "sst2",
                 "sts-b": "stsb"}


def canonical_glue_task_name(task_name):
    task_name = task_name.lower()
    canonical_name = _TASK_ALIASES.get(task_name, task_name)
    if canonical_name not in TASK_TO_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task_name}")
    return canonical_name


def get_glue_validation_split(task_name, matched=True):
    task_name = task_name.lower()
    canonical_name = canonical_glue_task_name(task_name)
    if task_name == "mnli-m":
        return "validation_matched"
    if task_name == "mnli-mm":
        return "validation_mismatched"
    if canonical_name == "mnli":
        return "validation_matched" if matched else "validation_mismatched"
    return "validation"
# ---------------------------------------------------------------------------


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def build_model(model_name, num_labels, cache_dir, lora_r, lora_alpha,
                lora_dropout):
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, cache_dir=cache_dir)
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r,
                             lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    return get_peft_model(base, peft_config)


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state, cur_round = ckpt["model"], ckpt.get("cur_round")
    else:
        state, cur_round = ckpt, None

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = sorted(trainable - set(state))
    if missing:
        raise RuntimeError(
            f"checkpoint is missing {len(missing)} trainable tensors, "
            f"e.g. {missing[:5]}")
    model.load_state_dict(state, strict=False)
    return state, cur_round, trainable


def split_fingerprint(eval_ds, header):
    h = hashlib.sha256()
    h.update(header.encode())
    for ex in eval_ds:
        ids = np.asarray(ex["input_ids"], dtype=np.int64).tobytes()
        lab = np.asarray([ex["label"]], dtype=np.int64).tobytes()
        h.update(ids)
        h.update(lab)
    return "sha256:" + h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    hf_home = os.path.expanduser(os.environ.get(
        "HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface")))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--model", default="FacebookAI/roberta-large")
    ap.add_argument("--num-labels", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--tok-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--cache-dir",
        default=os.path.expanduser(os.environ.get("MODEL_CACHE", os.path.join(hf_home, "hub"))))
    ap.add_argument("--data-root",
                    default=os.path.expanduser(os.environ.get("DATA_ROOT", os.environ.get(
                        "HF_DATASETS_CACHE", os.path.join(hf_home, "datasets")))))
    ap.add_argument("--glue-repo", default="nyu-mll/glue")
    ap.add_argument("--out", default=os.path.join(here, "metrics.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = canonical_glue_task_name(args.task)
    eval_split_name = get_glue_validation_split(args.task)

    # --- data (identical preprocessing to the framework dataloader) --------
    from datasets import load_dataset
    from transformers import AutoTokenizer

    raw = load_dataset(args.glue_repo, task, cache_dir=args.data_root)
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              cache_dir=args.cache_dir)

    def preprocess(examples):
        inp = ((examples[sentence1_key],) if sentence2_key is None
               else (examples[sentence1_key], examples[sentence2_key]))
        return tokenizer(*inp, padding="max_length",
                         max_length=args.tok_len, truncation=True)

    raw = raw.map(preprocess, batched=True, load_from_cache_file=True)
    raw.set_format(type="torch", columns=["input_ids", "attention_mask",
                                          "label"])
    eval_ds = raw[eval_split_name]

    # --- model + checkpoint -------------------------------------------------
    model = build_model(args.model, args.num_labels, args.cache_dir,
                        args.lora_r, args.lora_alpha, args.lora_dropout)
    state, cur_round, trainable = load_checkpoint(model, args.ckpt)

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    ys_pred, ys_true = [], []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            loss_sum += F.cross_entropy(logits, labels,
                                        reduction="sum").item()
            n += labels.shape[0]
            ys_pred.extend(preds.cpu().tolist())
            ys_true.extend(labels.cpu().tolist())

    from evaluate import load as load_metric
    glue_metric = load_metric("glue", task, trust_remote_code=True)
    metric = glue_metric.compute(predictions=ys_pred, references=ys_true)
    metric_name = "accuracy" if "accuracy" in metric else sorted(metric)[0]
    metric_value = float(metric[metric_name])
    argmax_acc = float(np.mean(np.asarray(ys_pred) == np.asarray(ys_true)))

    header = (f"{task}|{eval_split_name}|{args.model}|tok_len={args.tok_len}|"
              f"n={len(eval_ds)}")
    fingerprint = split_fingerprint(eval_ds, header)

    metrics = {
        "status": "ok",
        "task": task,
        "split": eval_split_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "num_examples": int(len(eval_ds)),
        "seed": args.seed,
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "code_hash": sha256_file(os.path.join(here, "eval_ckpt.py")),
        "split_fingerprint": fingerprint,
        "checkpoint_cur_round": cur_round,
        "model": {
            "type": args.model + "@huggingface_llm",
            "num_labels": args.num_labels,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "tok_len": args.tok_len,
            "batch_size": args.batch_size,
        },
        "num_trainable_tensors": len(trainable),
        "extra": {
            "glue_metric_raw": {k: (float(v) if isinstance(v, (int, float))
                                    else v) for k, v in metric.items()},
            "argmax_accuracy": argmax_acc,
            "val_loss_sum": loss_sum,
            "val_avg_loss": loss_sum / n if n else None,
            "run_log_key": f"val_{metric_name}",
        },
    }

    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
