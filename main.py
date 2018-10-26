# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import os
import time
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage


def add_gaussian_noise(inputs, std):
    if std > 0.0:
        stds = np.ones(inputs.shape) * std
        noise = np.random.normal(loc=0.0, scale=stds)
        inputs = inputs + noise
    return inputs


def epsilon_greedy(tensor, p, high, low):
    random_mask = np.random.binomial(1, p, tensor.shape[0])
    random_tensor = np.random.uniform(low, high, tensor.shape)
    tensor = random_mask * random_tensor + (1 - random_mask) * tensor
    return tensor


args = get_args()

if args.continuous:
    print("Using continuous control suite")
    if args.use_gaussian_noise:
        from configurations_continuous_gaussian import load_params
    elif args.use_uniform_noise:
        from configurations_continuous_uniform import load_params
    elif args.use_sparse_noise:
        from configurations_continuous_sparse import load_params
else:
    print("Using Atari suite")
    if args.use_gaussian_noise:
        from configurations_gaussian import load_params
    elif args.use_uniform_noise:
        from configurations_uniform import load_params
    elif args.use_sparse_noise:
        from configurations_sparse import load_params

if args.run_index is not None:
    load_params(args)

assert args.algo in ['a2c', 'ppo']

num_updates = int(args.num_frames) // args.num_steps // args.num_processes


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")
    print(args)

    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    for gamma in args.gamma:
        with open(args.log_dir + '/MSE_' + str(gamma) + '_monitor.csv', "wt") as monitor_file:
            monitor = csv.writer(monitor_file)
            monitor.writerow(['update', 'error', str(int(args.num_frames) // args.num_steps )])

    os.environ['OMP_NUM_THREADS'] = '1'

    print("Using env {}".format(args.env_name))

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    num_heads = len(args.gamma) if not args.reward_predictor else len(args.gamma) - 1

    if len(envs.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, num_heads=num_heads, hidden_size=args.hidden_size)
    else:
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space, num_heads=num_heads,
                                 reward_predictor=args.reward_predictor, use_s=args.use_s, use_s_a=args.use_s_a,
                                 use_s_a_sprime=args.use_s_a_sprime)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    lrs = [args.lr] * len(actor_critic.param_groups)

    if not args.reward_predictor:
        assert len(actor_critic.param_groups) == len(lrs)
        model_params = [{'params': model_p, 'lr': args.lr} for model_p, lr in zip(actor_critic.param_groups, lrs)]
    else:
        model_params = [{'params': model_p, 'lr': p_lr} for model_p, p_lr in zip(actor_critic.param_groups[:-1], lrs)]
        model_params.append({'params': actor_critic.param_groups[-1], 'lr': args.lr_rp})

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(model_params, args.lr, eps=args.eps, alpha=args.alpha)

    elif args.algo == 'ppo':
        optimizer = optim.Adam(model_params, args.lr, eps=args.eps)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size,
                              gamma=args.gamma, use_rp=args.reward_predictor)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs, obs_tensor):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            obs_tensor[:, :-shape_dim0] = obs_tensor[:, shape_dim0:]
        obs_tensor[:, -shape_dim0:] = obs

    obs = envs.reset()

    update_current_obs(obs, current_obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    advantages_list = []

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                      Variable(rollouts.states[step], volatile=True),
                                                                      Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            cpu_actions = add_gaussian_noise(cpu_actions, args.action_noise)

            # Obser reward and next obs
            obs, raw_reward, done, info = envs.step(cpu_actions)

            reward = np.copy(raw_reward)

            reward = add_gaussian_noise(reward, args.reward_noise)

            reward = epsilon_greedy(reward, args.reward_epsilon, args.reward_high, args.reward_low)

            raw_reward = torch.from_numpy(np.expand_dims(np.stack(raw_reward), 1)).float()

            episode_rewards += raw_reward

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs)

            if args.reward_predictor:
                r_hat = actor_critic.predict_reward(Variable(rollouts.observations[step], volatile=True),
                                                    action,
                                                    Variable(current_obs, volatile=True))
                p_hat = min(args.rp_burn_in, j) / args.rp_burn_in
                estimate_reward = (1 - p_hat) * reward + p_hat * r_hat.data.cpu()
                reward = torch.cat([reward, estimate_reward], dim=-1)
                value = torch.cat([r_hat, value], dim=-1).data
            else:
                value = value.data

            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value, reward, masks,
                            raw_reward)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        if args.reward_predictor:
            if args.use_s or args.use_s_a:
                r_hat = actor_critic.predict_reward(Variable(rollouts.observations[-1], volatile=True),
                                                    Variable(rollouts.actions[-1], volatile=True),
                                                    None).data
                next_value = torch.cat([r_hat, next_value], dim=-1)
            else:
                next_value = torch.cat([torch.zeros(list(next_value.size())[:-1] + [1]),
                                        next_value], dim=-1)

        rollouts.compute_returns(next_value, args.use_gae, args.tau)

        if args.algo in ['a2c']:

            batch_states = Variable(rollouts.states[0].view(-1, actor_critic.state_size))
            batch_masks = Variable(rollouts.masks[:-1].view(-1, 1))
            batch_obs = Variable(rollouts.observations[:-1].view(-1, *obs_shape))
            batch_actions = Variable(rollouts.actions.view(-1, action_shape))

            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(batch_obs,
                                                                                           batch_states,
                                                                                           batch_masks,
                                                                                           batch_actions)
            if args.reward_predictor:
                batch_obs_prime = Variable(rollouts.observations[1:].view(-1, *obs_shape))
                values = torch.cat([actor_critic.predict_reward(batch_obs, batch_actions, batch_obs_prime),
                                    values], dim=-1)
            returns_as_variable = Variable(rollouts.returns[:-1])

            batched_v_loss = 0

            values = values.view(returns_as_variable.size())
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = returns_as_variable - values
            value_loss = advantages.pow(2).sum(-1).mean()
            action_loss = -(Variable(advantages[:, :, -1].unsqueeze(-1).data) * action_log_probs).mean()
            if args.reward_predictor:
                rp_error = (values[:, :, 0].data - rollouts.raw_rewards).pow(2).mean()
                advantages_list.append([rp_error, advantages[:, :, -1].pow(2).mean().data.cpu().numpy()[0]])
            else:
                advantages_list.append(advantages[:, :, -1].pow(2).mean().data.cpu().numpy()[0])


            optimizer.zero_grad()
            (batched_v_loss + value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

            advantages = advantages[:, :, -1]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                data_generator = rollouts.feed_forward_generator(advantages,
                                                            args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ, observations_batch_prime, true_rewards_batch = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(observations_batch),
                                                                                                   Variable(states_batch),
                                                                                                   Variable(masks_batch),
                                                                                                   Variable(actions_batch))

                    if args.reward_predictor:
                        values = torch.cat([actor_critic.predict_reward(Variable(observations_batch),
                                                                        Variable(actions_batch),
                                                                        Variable(observations_batch_prime)),
                                            values], dim=-1)

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    td = (Variable(return_batch) - values).pow(2)

                    value_loss = td.sum(-1).mean()

                    if args.reward_predictor:
                        rp_error = (values[:, 0].data - true_rewards_batch).pow(2).mean()
                        advantages_list.append([rp_error, td[:, -1].mean(0).data.cpu().numpy()])
                    else:
                        advantages_list.append(td[:, -1].mean(0).data.cpu().numpy())

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()

                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, "
                  "entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]
                       ))

            if len(advantages_list) > 2:
                advantages_array = np.array(advantages_list).reshape(-1, len(args.gamma)).T
                for g, gamma in enumerate(args.gamma):
                    with open(args.log_dir + '/MSE_' + str(gamma) + '_monitor.csv', "a") as monitor_file:
                        monitor = csv.writer(monitor_file)
                        monitor.writerow([total_num_steps, np.mean(advantages_array[g])])

            advantages_list = []


if __name__ == "__main__":
    main()



