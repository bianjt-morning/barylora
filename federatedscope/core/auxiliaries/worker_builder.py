"""Client/server selection for the final FedAvg transport."""

from federatedscope.core.workers import Server, Client


def get_client_cls(cfg):
    if cfg.federate.method.lower() != "fedavg":
        raise ValueError("The paper release uses FedAvg transport")
    return Client


def get_server_cls(cfg):
    if cfg.federate.method.lower() != "fedavg":
        raise ValueError("The paper release uses FedAvg transport")
    return Server
