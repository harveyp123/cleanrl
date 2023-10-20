#!/bin/bash

models_num=2
detach=1
total_timesteps_list=(16000000)
T_list=(1 1 1 1)
alpha_list=(0 1 1 1 1)
student_timesteps_list=(100000 5000000 5000000 5000000 5000000 5000000)
distill_timesteps_list=(2000000 10000000 2000000 2000000 2000000 2000000 2000000)
lr_list=(0.00025)
student_lr_list=(0.001)
student_weight_decay_list=(0)
obs_num_list=(100)
threshold_list=(0)
seed_list=(3)
gpu_list=(0)
prefix='25.seed'
for i in ${!gpu_list[@]};do
    total_timesteps=${total_timesteps_list[i]}
    T=${T_list[i]}
    alpha=${alpha_list[i]}
    student_timesteps=${student_timesteps_list[i]}
    distill_timesteps=${distill_timesteps_list[i]}
    lr=${lr_list[i]}
    student_lr=${student_lr_list[i]}
    student_weight_decay=${student_weight_decay_list[i]}
    obs_num=${obs_num_list[i]}
    threshold=${threshold_list[i]}
    seed=${seed_list[i]}
    gpu=${gpu_list[i]}
    experiment_name=nonsymmetrized_${prefix}_total${total_timesteps}_T${T}_aplha${alpha}_detach${detach}_distill_steps${distill_timesteps}_student_steps${student_timesteps}_lr${lr}_student_lr${student_lr}_weight_decay${student_weight_decay}_obs_num${obs_num}_threshold${threshold}_seed${seed}_gpu${gpu}
    folder_name=logs/unequal_restart
    save=ckpt/unequal_restart/${experiment_name}.pt
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    ckpt_folder_name=ckpt/unequal_restart
    if [ ! -d ${ckpt_folder_name} ]; then
        mkdir -p ${ckpt_folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u cleanrl/ppo_atari_unequal_restart.py \
    --exp-name ${experiment_name} \
    --total-timesteps ${total_timesteps} \
    --T ${T} \
    --alpha ${alpha} \
    --student-alpha 1 \
    --student-timesteps ${student_timesteps} \
    --distill-timesteps ${distill_timesteps} \
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