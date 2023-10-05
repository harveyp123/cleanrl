export alpha=0.0
export n_agent=1


# export seed=1
# export seed=2
export seed=3


# export total_timesteps=10000000

# export total_timesteps=1000000
# export total_timesteps=2500000
# export total_timesteps=5000000
# export total_timesteps=15000000
export total_timesteps=21000000
mkdir -p ./log/consensus_total_timesteps_${total_timesteps}/




nohup python cleanrl/ppo_atari_consensus.py --env-id BreakoutNoFrameskip-v4 \
 --num-agent ${n_agent} --alpha-values ${alpha} --total-timesteps ${total_timesteps} --seed ${seed}\
 > ./log/consensus_total_timesteps_${total_timesteps}/ppo_atari_nagent_${n_agent}_alpha_${alpha}_seed_${seed}.txt &