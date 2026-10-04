import os

from transformers import AutoTokenizer
from datasets import load_dataset

task_to_keys = {
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

_task_aliases = {
    "mnli-m": "mnli",
    "mnli-mm": "mnli",
    "sst-2": "sst2",
    "sts-b": "stsb",
}


def canonical_glue_task_name(task_name):
    task_name = task_name.lower()
    canonical_name = _task_aliases.get(task_name, task_name)
    if canonical_name not in task_to_keys:
        supported = ", ".join(sorted(task_to_keys))
        raise ValueError(
            f"Unsupported GLUE task: {task_name}. Supported tasks: {supported}"
        )
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


def resolve_glue_dataset_repo():
    configured_repo = os.environ.get("FEDLORA_GLUE_REPO")
    if configured_repo:
        return configured_repo
    if os.environ.get("HF_ENDPOINT"):
        return "nyu-mll/glue"
    return "glue"


def load_glue_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    raw_task_name, _ = config.data.type.split('@')
    task_name = canonical_glue_task_name(raw_task_name)
    
    # download the dataset.
    datasets = load_dataset(
        resolve_glue_dataset_repo(),
        task_name,
        cache_dir=config.data.root,
    )
    
    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
        config.data.label_list = label_list    # added by me, update the config object of the label list
    else:
        num_labels = 1
    config.data.num_labels = num_labels    # added by me, update the config object of the number of labels
    
    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=config.llm.cache.model
    )
    
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=config.llm.tok_len, truncation=True)
        return result
    
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    train_dataset = datasets["train"]
    eval_split = get_glue_validation_split(
        raw_task_name,
        matched=config.data.matched,
    )
    eval_dataset = datasets[eval_split]
    # test_dataset = datasets["test_matched" if config.data.matched else "test_mismatched"]
    
    dataset = (train_dataset, eval_dataset, [])

    return dataset, config
