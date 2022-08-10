import torch
import logging

LOGGER = logging.getLogger(__name__)

def create(conf, optimizer):
    scheduler = None
    if conf['type'] == 'CosineAnnealingLR':
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **conf['params'])
    else:
        raise AttributeError(f'not support scheduler config: {conf}')

    scheduler = scheduler_lr

    return scheduler