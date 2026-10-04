"""FedAvg transport for ordinary trainable task-head parameters."""
from federatedscope.core.aggregators import ClientsAvgAggregator, OnlineClientsAvgAggregator


def get_aggregator(method, model=None, device=None, online=False, config=None):
    if method.lower() != 'fedavg' or config.backend != 'torch':
        raise ValueError('The paper release uses PyTorch FedAvg transport')
    if config.aggregator.robust_rule != 'fedavg':
        raise ValueError('Only fedavg head aggregation is included')
    if online:
        return OnlineClientsAvgAggregator(
            model=model, device=device, config=config,
            src_device=device if config.federate.share_local_model else 'cpu')
    return ClientsAvgAggregator(model=model, device=device, config=config)
