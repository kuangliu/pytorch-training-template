'''Checkpoint util.'''

import os
import torch
import utils.distributed as dist


def save_checkpoint(model, optimizer, epoch, cfg):
    '''Save checkpoint.

    Args:
      save_path (str): save file path.
      model (model): model to save.
      optimizer (optim): optimizer to save.
      epoch (int): current epoch index.
      cfg (CfgNode): configs to save.
    '''
    # Only save on master process.
    if not dist.is_master_proc(cfg.NUM_GPUS):
        return

    state_dict = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()

    checkpoint = {
        'epoch': epoch,
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'cfg': cfg.dump(),
    }

    if not os.path.isdir(cfg.TRAIN.CHECKPOINT_DIR):
        os.mkdir(cfg.TRAIN.CHECKPOINT_DIR)
    save_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, '%d.pth' % epoch)
    torch.save(checkpoint, save_path)
