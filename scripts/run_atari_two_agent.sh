#!/bin/bash

models_num=2
detach=1
total_timesteps_list=(1000000)
T_list=(1 1 1 1)
alpha_list=(1 1 1 1)
student_alpha_list=(1 1 1 1)
student_timesteps_list=(0 5000000 5000000 5000000 5000000)
distill_timesteps_list=(1000000 2000000 2000000 2000000 2000000)
gpu_list=(0)
prefix='1.debug'
for i in ${!gpu_list[@]};do
    total_timesteps=${total_timesteps_list[i]}
    T=${T_list[i]}
    alpha=${alpha_list[i]}
    student_alpha=${student_alpha_list[i]}
    student_timesteps=${student_timesteps_list[i]}
    distill_timesteps=${distill_timesteps_list[i]}
    gpu=${gpu_list[i]}
    experiment_name=nonsymmetrized_${prefix}_total${total_timesteps}_T${T}_aplha${alpha}_student_alpha${student_alpha}_detach${detach}_distill_steps${distill_timesteps}_student_steps${student_timesteps}_gpu${gpu}
    folder_name=logs/two_agent
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u cleanrl/ppo_atari_two_agent.py \
    --exp-name ${experiment_name} \
    --total-timesteps ${total_timesteps} \
    --T ${T} \
    --alpha ${alpha} \
    --student-alpha ${student_alpha} \
    --student-timesteps ${student_timesteps} \
    --distill-timesteps ${distill_timesteps} \
    --gpu ${gpu} \
    --num-agent ${models_num} \
    --detach ${detach} \
    --seed 0 \
    > ${log_filename} 2>&1 &
done