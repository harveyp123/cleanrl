#!/bin/bash
experiment_name='test'
total_timesteps=20000000
env_id_list=(DemonAttackNoFrameskip-v4 RobotankNoFrameskip-v4 UpNDownNoFrameskip-v4 ZaxxonNoFrameskip-v4 GravitarNoFrameskip-v4)
env_id='DemonAttackNoFrameskip-v4'
folder_name=plot/${env_id}/total_timesteps${total_timesteps}
if [ ! -d ${folder_name} ]; then
    mkdir -p ${folder_name}
fi
log_filename=${folder_name}/${experiment_name}.log
nohup python -u plot/unequal_restart.py \
    --env_id ${env_id} \
    --extract wandb \
    --total_timesteps ${total_timesteps} \
    --wandb_key ca2f2a2ae6e84e31bbc09a8f35f9b9a534dfbe9b \
    --wandb_user jincan333 \
    --wandb_project ensemble_distill_atari \
> ${log_filename} 2>&1 &