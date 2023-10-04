# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import ensemble_consensus_util as esb_util



from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--learning-rate-min", type=float, default=2.5e-6,
        help="the minimum learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-agent", type=int, default=2,
        help="the number of parallel game agent")
    parser.add_argument("--alpha-values", type=float, default=0,
        help="distillation strength")
    parser.add_argument("--test-ensemble", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Test the ensemble agent or not")
    parser.add_argument("--smooth-return", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Smooth the return or not")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), logits


class Agent_ensemble(nn.Module):
    def __init__(self, agent_list):
        super().__init__()
        self.agent_list = agent_list
        self.num_models = len(agent_list)

    def get_action(self, x, action=None):

        # print("x shape:", x.shape)

        logits = []
        for i in range(self.num_models):
            hidden = self.agent_list[i].network(x / 255.0)
            logits.append(self.agent_list[i].actor(hidden))
        # print("original logits shape:", logits[0].shape)
        logits = esb_util.reduce_ensemble_logits(logits)

        # print("logits shape:", logits.shape)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__alpha{args.alpha_values}__seed{args.seed}__{int(time.time())}"
    writer_list = []


    #### number of update epochs
    num_updates = args.total_timesteps // args.batch_size

    for i in range(args.num_agent):
        
        writer = SummaryWriter(f"runs/{args.exp_name}_total_time_{args.total_timesteps}/{run_name}/agent_{i}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        writer_list.append(writer)
    if args.test_ensemble:
        writer_ensemble = SummaryWriter(f"runs/{args.exp_name}_total_time_{args.total_timesteps}/{run_name}/ensemble")
        writer_ensemble.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")




    # env setup
    envs_list = []
    agent_list = []
    optimizer_list = []
    lr_scheduler_list = []
    

    obs_list = []
    actions_list = []
    logprobs_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    finished_frames_list = []
    finished_runs_list = []
    avg_return_list = []
    avg_length_list = []

    next_obs_list = []
    next_done_list = []
    for i in range(args.num_agent):
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        envs_list.append(envs)

        agent = Agent(envs).to(device)
        agent_list.append(agent)

        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        optimizer_list.append(optimizer)
        lr_scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = num_updates, 
                        eta_min = args.learning_rate_min))

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        obs_list.append(obs)
        actions_list.append(actions)
        logprobs_list.append(logprobs)
        rewards_list.append(rewards)
        dones_list.append(dones)
        values_list.append(values)

        # TRY NOT TO MODIFY: variables necessary to record the game while starting the game
        
        finished_runs = 0
        finished_frames = 0
        avg_return = 0.0
        avg_length = 0.0
        
        finished_runs_list.append(finished_runs)
        finished_frames_list.append(finished_frames)
        avg_return_list.append(avg_return)
        avg_length_list.append(avg_length)

        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        
        next_obs_list.append(next_obs)
        next_done_list.append(next_done)


    consensus_model = esb_util.ClassifierConsensusForthLoss(args)
    optimizer_consensus = optim.Adam(consensus_model.parameters(), lr=args.learning_rate, eps=1e-5)
    lr_scheduler_consensus = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_consensus, T_max = num_updates, 
                        eta_min = args.learning_rate_min)

    ######## introduce an ensemble agent to play the game ########
    if args.test_ensemble:
        # env setup
        envs_ensemble = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
        assert isinstance(envs_ensemble.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent_ensemble = Agent_ensemble(agent_list)
        next_obs_ensemble = torch.Tensor(envs_ensemble.reset()).to(device)
        finished_runs_ensemble = 0.0
        finished_frames_ensemble = 0.0
        avg_return_ensemble = 0.0
        avg_length_ensemble = 0.0

    global_step = 0
    
        
    start_time = time.time()



    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            for i in range(args.num_agent):
                obs_list[i][step] = next_obs_list[i]
                dones_list[i][step] = next_done_list[i]

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _ = agent_list[i].get_action_and_value(next_obs_list[i])
                    values_list[i][step] = value.flatten()
                actions_list[i][step] = action
                logprobs_list[i][step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs_list[i].step(action.cpu().numpy())
                rewards_list[i][step] = torch.tensor(reward).to(device).view(-1)
                next_obs_list[i], next_done_list[i] = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                
                for item in info:
                    finished_frames_list[i] += 1
                    if "episode" in item.keys():
                        finished_runs_list[i] += 1
                        print(f"Agent {i} play result: finished_runs={finished_runs_list[i]}, episodic_return={item['episode']['r']}")
                        if args.smooth_return:
                            avg_return_list[i] = 0.9 * avg_return_list[i] + 0.1 * item["episode"]["r"]
                            avg_length_list[i] = 0.9 * avg_length_list[i] + 0.1 * item["episode"]["l"]
                        else:
                            avg_return_list[i] = item["episode"]["r"]
                            avg_length_list[i] = item["episode"]["l"]
                        # writer_list[i].add_scalar("charts/episodic_return", avg_return_list[i], finished_runs_list[i])
                        # writer_list[i].add_scalar("charts/episodic_length", avg_length_list[i], finished_runs_list[i])
                        writer_list[i].add_scalar("charts/episodic_return", avg_return_list[i], finished_frames_list[i])
                        writer_list[i].add_scalar("charts/episodic_length", avg_length_list[i], finished_frames_list[i])

                        # break
            ######## introduce an ensemble agent to play the game ########
            if args.test_ensemble:
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action = agent_ensemble.get_action(next_obs_ensemble)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs_ensemble, reward, done, info = envs_ensemble.step(action.cpu().numpy())
                next_obs_ensemble= torch.Tensor(next_obs_ensemble).to(device)

                for item in info:
                    finished_frames_ensemble += 1
                    if "episode" in item.keys():
                        finished_runs_ensemble += 1
                        if args.smooth_return:
                            avg_return_ensemble = 0.9 * avg_return_ensemble + 0.1 * item["episode"]["r"]
                            avg_length_ensemble = 0.9 * avg_length_ensemble + 0.1 * item["episode"]["l"]
                        else: 
                            avg_return_ensemble = item["episode"]["r"]
                            avg_length_ensemble = item["episode"]["l"]
                        print(f"Ensemble agent play result: finished_runs={finished_runs_ensemble}, episodic_return={item['episode']['r']}")
                        # writer_ensemble.add_scalar("charts/episodic_return", avg_return_ensemble, finished_runs_ensemble)
                        # writer_ensemble.add_scalar("charts/episodic_length", avg_length_ensemble, finished_runs_ensemble)
                        writer_ensemble.add_scalar("charts/episodic_return", avg_return_ensemble, finished_frames_ensemble)
                        writer_ensemble.add_scalar("charts/episodic_length", avg_length_ensemble, finished_frames_ensemble)
                        
                        # break


        advantages_list = []
        returns_list = []
        # bootstrap value if not done
        with torch.no_grad():
            for i in range(args.num_agent):
                next_value = agent_list[i].get_value(next_obs_list[i]).reshape(1, -1)
                advantages = torch.zeros_like(rewards_list[i]).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_list[i]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_list[i][t + 1]
                        nextvalues = values_list[i][t + 1]
                    delta = rewards_list[i][t] + args.gamma * nextvalues * nextnonterminal - values_list[i][t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_list[i]
                advantages_list.append(advantages)
                returns_list.append(returns)

        b_obs_list = []
        b_logprobs_list = []
        b_actions_list = []
        b_advantages_list = []
        b_returns_list = []
        b_values_list = []
        for i in range(args.num_agent):
            # flatten the batch
            b_obs = obs_list[i].reshape((-1,) + envs_list[i].single_observation_space.shape)
            b_logprobs = logprobs_list[i].reshape(-1)
            b_actions = actions_list[i].reshape((-1,) + envs_list[i].single_action_space.shape)
            b_advantages = advantages_list[i].reshape(-1)
            b_returns = returns_list[i].reshape(-1)
            b_values = values_list[i].reshape(-1)

            b_obs_list.append(b_obs)
            b_logprobs_list.append(b_logprobs)
            b_actions_list.append(b_actions)
            b_advantages_list.append(b_advantages)
            b_returns_list.append(b_returns)
            b_values_list.append(b_values)

        # Optimizing the policy and value network
        b_inds_list = []
        clipfracs_list = []
        for i in range(args.num_agent):
            b_inds = np.arange(args.batch_size)
            b_inds_list.append(b_inds)
            clipfracs = []
            clipfracs_list.append(clipfracs)

        for epoch in range(args.update_epochs):
            for i in range(args.num_agent):
                np.random.shuffle(b_inds_list[i])
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size

                mb_inds_list = []
                for i in range(args.num_agent):
                    mb_inds = b_inds[start:end]
                    mb_inds_list.append(mb_inds)

                entropy_list = []
                newvalue_list = []
                logratio_list = []
                ratio_list = []
                logits_self_list = []
                ##### logits_other_list gives detached logits from all agents on all sets of b_obs #####
                logits_other_list = []
                for i in range(args.num_agent):

                    ##### logits_other gives detached logits from all agents on current b_obs #####
                    logits_other = []
                    ####### Pass the env observation to self agent
                    for j in range(args.num_agent):
                        
                        if i == j:
                            _, newlogprob, entropy, newvalue, logits = agent_list[j].get_action_and_value(b_obs_list[i][mb_inds_list[i]], b_actions_list[i].long()[mb_inds_list[i]])
                            logratio = newlogprob - b_logprobs_list[i][mb_inds_list[i]]
                            ratio = logratio.exp()

                            entropy_list.append(entropy)
                            newvalue_list.append(newvalue)
                            logratio_list.append(logratio)
                            ratio_list.append(ratio)
                            logits_self_list.append(logits)
                            
                            logits_other.append(logits.detach())
                        else: 
                            with torch.no_grad():
                                _, _, _, _, logits = agent_list[j].get_action_and_value(b_obs_list[i][mb_inds_list[i]], b_actions_list[i].long()[mb_inds_list[i]])
                                logits_other.append(logits)
                    logits_other_list.append(logits_other)

                old_approx_kl_list = []
                approx_kl_list = []

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    for i in range(args.num_agent):
                        old_approx_kl = (-logratio_list[i]).mean()
                        approx_kl = ((ratio_list[i] - 1) - logratio_list[i]).mean()
                        clipfracs_list[i] += [((ratio_list[i] - 1.0).abs() > args.clip_coef).float().mean().item()]
                        
                        old_approx_kl_list.append(old_approx_kl)
                        approx_kl_list.append(approx_kl)
                
                loss = 0

                
                v_loss_list = []
                
                pg_loss_list = []
                entropy_loss_list = []
                
                for i in range(args.num_agent):
                    mb_advantages = b_advantages_list[i][mb_inds_list[i]]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio_list[i]
                    pg_loss2 = -mb_advantages * torch.clamp(ratio_list[i], 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue_list[i].view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns_list[i][mb_inds_list[i]]) ** 2
                        v_clipped = b_values_list[i][mb_inds_list[i]] + torch.clamp(
                            newvalue - b_values_list[i][mb_inds_list[i]],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns_list[i][mb_inds_list[i]]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns_list[i][mb_inds_list[i]]) ** 2).mean()

                    entropy_loss = entropy_list[i].mean()
                    loss += pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    v_loss_list.append(v_loss.item())
                    pg_loss_list.append(pg_loss.item())
                    entropy_loss_list.append(entropy_loss.item())

                loss += consensus_model(logits_self_list, logits_other_list)
                ##### Compute final gradient
                for i in range(args.num_agent):
                    optimizer_list[i].zero_grad()
                optimizer_consensus.zero_grad()
                loss.backward()
                for i in range(args.num_agent):
                    nn.utils.clip_grad_norm_(agent_list[i].parameters(), args.max_grad_norm)
                    optimizer_list[i].step()
                optimizer_consensus.step()

            ##### Disable this function since now we have multiple agent
            # if args.target_kl is not None:
            #     if approx_kl > args.target_kl:
            #         break


        for i in range(args.num_agent):
            y_pred, y_true = b_values_list[i].cpu().numpy(), b_returns_list[i].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer_list[i].add_scalar("charts/learning_rate", optimizer_list[i].param_groups[0]["lr"], global_step)
            writer_list[i].add_scalar("losses/value_loss", v_loss_list[i], global_step)
            writer_list[i].add_scalar("losses/policy_loss", pg_loss_list[i], global_step)
            writer_list[i].add_scalar("losses/entropy", entropy_loss_list[i], global_step)
            writer_list[i].add_scalar("losses/old_approx_kl", old_approx_kl_list[i].item(), global_step)
            writer_list[i].add_scalar("losses/approx_kl", approx_kl_list[i].item(), global_step)
            writer_list[i].add_scalar("losses/clipfrac", np.mean(clipfracs_list[i]), global_step)
            writer_list[i].add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer_list[i].add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # Annealing the rate if instructed to do so.

        if args.anneal_lr:
            ######## Original anneal
            # frac = 1.0 - (update - 1.0) / num_updates
            # lrnow = frac * args.learning_rate
            # for i in range(args.num_agent):
            #     optimizer_list[i].param_groups[0]["lr"] = lrnow
            # optimizer_consensus.param_groups[0]["lr"] = lrnow
            
            ####### Cosine anneal
            for lr_scheduler in lr_scheduler_list:
                lr_scheduler.step()
            lr_scheduler_consensus.step()
    for i in range(args.num_agent):
        envs_list[i].close()
        writer_list[i].close()
    print("Finished all runs")