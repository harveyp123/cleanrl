#!/bin/bash

models_num=2
detach=1
total_timesteps_list=(20000000)
T_list=(1 1 1 1)
alpha_list=(0.1 1 1 1 1)
student_ratio=(5 5 5 5)
obs_num_list=(10 10 10 10)
lr_list=(0.00025)
student_lr_list=(0.00025)
student_weight_decay_list=(0)
threshold_list=(0)
seed_list=(1)
gpu_list=(0)
env_id='ZaxxonNoFrameskip-v4'
prefix='0.ours'
for i in ${!gpu_list[@]};do
    total_timesteps=${total_timesteps_list[i]}
    T=${T_list[i]}
    alpha=${alpha_list[i]}
    student_ratio=${student_ratio[i]}
    lr=${lr_list[i]}
    student_lr=${student_lr_list[i]}
    student_weight_decay=${student_weight_decay_list[i]}
    obs_num=${obs_num_list[i]}
    threshold=${threshold_list[i]}
    seed=${seed_list[i]}
    gpu=${gpu_list[i]}
    experiment_name=${env_id}_${prefix}_total${total_timesteps}_aplha${alpha}_student_ratio${student_ratio}_student_lr${student_lr}_weight_decay${student_weight_decay}_obs_num${obs_num}_threshold${threshold}_seed${seed}
    folder_name=logs/unequal_steps_${env_id}
    save=ckpt/unequal_steps_${env_id}/${experiment_name}.pt
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    ckpt_folder_name=ckpt/unequal_steps_${env_id}
    if [ ! -d ${ckpt_folder_name} ]; then
        mkdir -p ${ckpt_folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u cleanrl/ppo_atari_unequal_steps.py \
    --env-id ${env_id} \
    --exp-name ${experiment_name} \
    --total-timesteps ${total_timesteps} \
    --T ${T} \
    --alpha ${alpha} \
    --student-alpha 1 \
    --student-ratio ${student_ratio} \
    --learning-rate ${lr} \
    --student-lr ${student_lr} \
    --student-weight-decay ${student_weight_decay} \
    --obs-num ${obs_num} \
    --threshold ${threshold} \
    --gpu ${gpu} \
    --num-agent ${models_num} \
    --detach ${detach} \
    --seed ${seed} \
    --save ${save} \
    > ${log_filename} 2>&1 &
done