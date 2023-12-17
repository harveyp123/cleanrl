#!/bin/bash
experiment_name='result'
total_timesteps=20000000
env_id_list=(GravitarNoFrameskip-v4)
# env_id_list=(UpNDownNoFrameskip-v4 GravitarNoFrameskip-v4 BreakoutNoFrameskip-v4 BeamRiderNoFrameskip-v4 ZaxxonNoFrameskip-v4)
for i in ${!env_id_list[@]};do
    env_id=${env_id_list[i]}
    folder_name=plots/${env_id}/total_timesteps${total_timesteps}
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u plots/unequal_steps.py \
        --env_id ${env_id} \
        --extract wandb \
        --total_timesteps ${total_timesteps} \
        --wandb_key ca2f2a2ae6e84e31bbc09a8f35f9b9a534dfbe9b \
        --wandb_user jincan333 \
        --wandb_project ensemble_distill_atari \
    > ${log_filename} 2>&1 &
done