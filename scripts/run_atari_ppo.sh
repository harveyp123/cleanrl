mkdir -p ./log/
# python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4
nohup python cleanrl/ppo_atari_consensus.py --env-id BreakoutNoFrameskip-v4 > ./log/ppo_atari_avg.txt &