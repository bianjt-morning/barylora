import torch
from federatedscope.glue.model.adapter_builder import AdapterModel


def get_model_from_huggingface(model_name, config):
    from transformers import AutoModelForSequenceClassification

    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model
    
    # added by me, for GLUE
    kwargs['num_labels'] = config.data.num_labels

    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


def get_llm(config):    
    model_name, model_hub = config.model.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    
    return model
