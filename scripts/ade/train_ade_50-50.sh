#!/bin/bash

GPU=0,1,2,3
BS=6  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='50-50'
EPOCH=100
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0 # 300 for Ours-M

NAME='DKD'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_ade.py -d 0 --test -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_2/checkpoint-epoch${EPOCH}.pth
