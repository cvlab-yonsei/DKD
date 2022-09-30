import argparse
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from trainer.trainer_ade import Trainer_base, Trainer_incremental
from utils.parse_config import ConfigParser
from logger.logger import Logger

torch.backends.cudnn.benchmark = True


def main(config):
    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        # Single node, mutliple GPUs
        config.config['world_size'] = ngpus_per_node * config['world_size']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Rather using distributed, use DataParallel
        main_worker(None, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    if config['multiprocessing_distributed']:
        config.config['rank'] = config['rank'] * ngpus_per_node + gpu

    dist.init_process_group(
        backend=config['dist_backend'], init_method=config['dist_url'],
        world_size=config['world_size'], rank=config['rank']
    )
    
    # Set looging
    rank = dist.get_rank()
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f'train(rank{rank})', verbosity=2)

    # fix random seeds for reproduce
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Task information
    task_step = config['data_loader']['args']['task']['step']

    # Create Dataloader
    dataset = config.init_obj('data_loader', module_data)

    test_loader = dataset.get_test_loader()

    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes = []
    for i in range(1, task_step + 1):
        c, _ = dataset.get_task_labels(step=i)
        new_classes += c
    logger.info(f"Old Classes: {old_classes}")
    logger.info(f"New Classes: {new_classes}")

    # Create Model
    model = config.init_obj('arch', module_arch, **{"classes": dataset.get_per_task_classes()})

    # Convert BN to SyncBN
    if config['multiprocessing_distributed'] and (config['arch']['args']['norm_act'] == 'bn_sync'):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)

    evaluator_test = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes + 1, list(set(old_classes + [0])), new_classes]
    )

    trainer = Trainer_base(
        model=model,
        optimizer=None,
        evaluator=(None, evaluator_test),
        config=config,
        task_info=dataset.task_info(),
        data_loader=(None, None, test_loader),
        lr_scheduler=None,
        logger=logger, gpu=gpu,
    )

    torch.distributed.barrier()
    trainer.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Class incremental Semantic Segmentation')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type action target', defaults=(None, float, None, None))
    options = [
        CustomArgs(['--multiprocessing_distributed'], action='store_true', target='multiprocessing_distributed'),
        CustomArgs(['--dist_url'], type=str, target='dist_url'),

        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),
        
        CustomArgs(['--test'], action='store_false', target='test'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
