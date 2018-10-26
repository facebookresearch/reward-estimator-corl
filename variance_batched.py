# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import numpy as np
import pickle
import torch
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
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

if args.use_gaussian_noise:
    from configurations_continuous_gaussian import load_params
elif args.use_uniform_noise:
    from configurations_continuous_uniform import load_params
elif args.use_sparse_noise:
    from configurations_continuous_sparse import load_params



def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")
    print(args)

    load_params(args)

    if not args.reward_predictor:
        return

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['OMP_NUM_THREADS'] = '1'

    print("Using env {}".format(args.env_name))

    envs = [make_env(args.env_name, args.seed, i, None)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    save_path = os.path.join(args.save_dir, args.algo)

    actor_critic = torch.load(os.path.join(save_path, args.env_name + ".pt"))[0]

    if args.cuda:
        actor_critic.cuda()

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size,
                              gamma=args.gamma, use_rp=args.reward_predictor, use_delta=args.use_delta)
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

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    true_rewards = [[] for _ in range(100)]
    noisy_rewards = [[] for _ in range(100)]
    estimate_rewards = [[] for _ in range(100)]

    for j in range(100):
        end = False
        while not end:
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(Variable(current_obs, volatile=True), None, None)
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, raw_reward, done, info = envs.step(cpu_actions)
            true_rewards[j].append(raw_reward[0])

            reward = np.copy(raw_reward)

            reward = add_gaussian_noise(reward, args.reward_noise)

            reward = epsilon_greedy(reward, args.reward_epsilon, args.reward_high, args.reward_low)

            noisy_rewards[j].append(reward[0])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            past_obs = current_obs.clone()

            update_current_obs(obs, current_obs)

            r_hat = actor_critic.predict_reward(Variable(past_obs, volatile=True),
                                                action,
                                                Variable(current_obs, volatile=True))

            estimate_rewards[j].append(r_hat.data.cpu().numpy()[0])

            if done[0]:
                end = True
    with open(args.log_dir + '/true_rewards.pkl', "wb") as output_file:
        pickle.dump(true_rewards, output_file)
    with open(args.log_dir + '/noisy_rewards.pkl', "wb") as output_file:
        pickle.dump(noisy_rewards, output_file)
    with open(args.log_dir + '/estimate_rewards.pkl', "wb") as output_file:
        pickle.dump(estimate_rewards, output_file)








if __name__ == "__main__":
    main()
