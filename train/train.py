import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import glog as log
import utils.multiprocessing as mp

from config.defaults import get_cfg
from models.model import build_model
from models.optimizer import construct_optimizer
from datasets.loader import construct_loader, shuffle_dataset

from utils.metrics import topks_correct
from utils.checkpoint import save_checkpoint
from utils.optim_util import set_lr, get_epoch_lr
from utils.distributed import all_reduce, all_gather, is_master_proc


def train_epoch(train_loader, model, optimizer, epoch, cfg):
    '''Epoch training.

    Args:
      train_loader (DataLoader): training data loader.
      model (model): the video model to train.
      optimizer (optim): the optimizer to perform optimization on the model's parameters.
      epoch (int): current epoch of training.
      cfg (CfgNode): configs. Details can be found in config/defaults.py
    '''
    if is_master_proc():
        log.info('Epoch: %d' % epoch)

    model.train()
    num_batches = len(train_loader)
    train_loss = train_acc = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda()

        # Update lr.
        lr = get_epoch_lr(cfg, epoch+float(batch_idx)/num_batches)
        set_lr(optimizer, lr)

        # Forward.
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, reduction='mean')

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy.
        acc = topk_accuracies(outputs, labels, (1,))

        # Gather all predictions across all devices.
        if cfg.NUM_GPUS > 1:
            loss, acc = all_reduce([loss, acc[0]])
        else:
            acc = acc[0]

        if is_master_proc():
            train_loss += loss.item()
            train_acc += acc.item()
            log.info('Loss: %.3f | Acc: %.2f | LR: %.3f' %
                     (train_loss/(batch_idx+1), train_acc / (batch_idx+1), lr))


@torch.no_grad()
def eval_epoch(val_loader, model, epoch, cfg):
    '''Evaluate the model on the val set.

    Args:
      val_loader (loader): data loader to provide validation data.
      model (model): model to evaluate the performance.
      epoch (int): number of the current epoch of training.
      cfg (CfgNode): configs. Details can be found in config/defaults.py
    '''
    if is_master_proc():
        log.info('Testing..')

    model.eval()
    test_loss = 0.0
    correct = total = 0.0
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, reduction='mean')

        # Gather all predictions across all devices.
        if cfg.NUM_GPUS > 1:
            loss = all_reduce([loss])[0]
            outputs, labels = all_gather([outputs, labels])

        # Accuracy.
        batch_correct = topks_correct(outputs, labels, (1,))[0]
        correct += batch_correct.item()
        total += labels.size(0)

        if is_master_proc():
            test_loss += loss.item()
            test_acc = correct / total
            log.info('Loss: %.3f | Acc: %.2f' %
                     (test_loss/(batch_idx+1), test_acc))


def train(cfg):
    train_loader = construct_loader(cfg, train=True)
    val_loader = construct_loader(cfg, train=False)

    model = build_model(cfg)
    optimizer = construct_optimizer(model, cfg)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        shuffle_dataset(train_loader, epoch)
        train_epoch(train_loader, model, optimizer, epoch, cfg)
        eval_epoch(val_loader, model, epoch, cfg)
        save_checkpoint(model, optimizer, epoch, cfg)


if __name__ == '__main__':
    cfg = get_cfg()

    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mp.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                train,
                'tcp://localhost:9999',
                0,  # shard_id
                1,  # num_shards
                'nccl',
                cfg,
            ),
            daemon=False,
        )
    else:
        train(cfg)
