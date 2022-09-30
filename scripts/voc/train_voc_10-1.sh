#!/bin/bash

PORT='tcp://127.0.0.1:12345'
GPU=1,2,3,4
BS=8  # Total 32
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='10-1'
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=100  # 100 for DKD-M

NAME='DKD'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 6 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 7 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 8 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 9 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 10 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 --test -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_10/checkpoint-epoch${EPOCH}.pth
