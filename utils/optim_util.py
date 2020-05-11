"""Learning rate policy."""

import math


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return (
        cfg.SOLVER.BASE_LR
        * (math.cos(math.pi * cur_epoch / cfg.TRAIN.MAX_EPOCH) + 1.0)
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind] * cfg.SOLVER.BASE_LR


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.TRAIN.STEPS + [cfg.TRAIN.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


def get_epoch_lr(cfg, cur_epoch):
    '''Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.

    Args:
      cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
      cur_epoch (float): current epoch index.
    '''
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
        lr_start = cfg.SOLVER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
            cfg, cfg.SOLVER.WARMUP_EPOCHS)
        alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def set_lr(optimizer, new_lr):
    '''Sets the optimizer lr to the specified value.

    Args:
      optimizer (optim): the optimizer using to optimize the current network.
      new_lr (float): the new learning rate to set.
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


if __name__ == '__main__':
    from config.defaults import get_cfg
    cfg = get_cfg()
    for epoch in range(300):
        lr = get_epoch_lr(cfg, epoch)
        print(epoch, lr)
