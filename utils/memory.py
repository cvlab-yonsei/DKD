"""
We modified the code from SSUL

SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data

from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader.task import get_task_labels, get_per_task_classes
from trainer.trainer_voc import Trainer_base, Trainer_incremental


def _prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
    

def memory_sampling_balanced(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()
    
    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

                outputs, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1
                
                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)
            
            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = 0
    
    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem

                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if memory_size <= num_sampled:
                break
        
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    
    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }
    
    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()
