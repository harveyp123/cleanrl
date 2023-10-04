

export total_timesteps=10000000
mkdir -p ./log/consensus_total_timesteps_${total_timesteps}/

# export seed=1
# export seed=2
# export seed=3
# export seed=4
export seed=5

# export n_agent=2
# export alpha=0.0

# export n_agent=2
# export alpha=1.0

export n_agent=2
export alpha=2.0



nohup python cleanrl/ppo_atari_consensus.py --env-id BreakoutNoFrameskip-v4 \
 --num-agent ${n_agent} --alpha-values ${alpha} --total-timesteps ${total_timesteps} --seed ${seed}\
 > ./log/consensus_total_timesteps_${total_timesteps}/ppo_atari_nagent_${n_agent}_alpha_${alpha}_seed_${seed}.txt &