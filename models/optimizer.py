import torch
import torch.nn as nn


def construct_optimizer(model, cfg):
    '''Construct a SGD or Adam optimizer.

    Args:
      model (model): model.
      cfg (config): configs.
    '''
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    bn_params = []
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if 'bn' in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)

    optim_params = [
        {'params': bn_params, 'weight_decay': cfg.SOLVER.BN_WEIGHT_DECAY},
        {'params': non_bn_parameters, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
    ]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(
        non_bn_parameters) + len(bn_params)

    if cfg.SOLVER.OPTIMIZING_METHOD == 'sgd':
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'adam':
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            'Does not support %s optimizer' % cfg.SOLVER.OPTIMIZING_METHOD)
