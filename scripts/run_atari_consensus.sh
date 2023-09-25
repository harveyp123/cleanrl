mkdir -p ./log/consensus/
# export n_agent=2
# export alpha=1.0

export n_agent=2
export alpha=2.0
nohup python cleanrl/ppo_atari_consensus.py --env-id BreakoutNoFrameskip-v4 \
 --num-agent ${n_agent} --alpha-values ${alpha} \
 --test-ensemble True > ./log/consensus/ppo_atari_nagent_${n_agent}_alpha_${alpha}.txt &