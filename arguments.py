# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, nargs="*", default=[0.99],
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='./logs/',
                        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--run-index', type=int, default=None,
                        help='run index for batched run')
    parser.add_argument('--name', default='',
                        help='name for config')
    parser.add_argument('--use-biased-replay', action='store_true', default=False,
                        help='use 50/50 reward sampling for first head')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for biased replay')
    parser.add_argument('--reward-predictor', action='store_true', default=False,
                        help='use model rewards')
    parser.add_argument('--reward-noise', type=float, default=0.0,
                        help='add noise to reward')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='use continuous control experiment suite')
    parser.add_argument('--lr-rp', type=float, default=0.0001,
                        help='learning rate for reward predictor')
    parser.add_argument('--rp-burn-in', type=int, default=25000,
                        help='number of updates to decay between using true reward and estimate')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='size of hidden layer')
    parser.add_argument('--state-noise', type=float, default=0.0,
                        help='add noise to the observations')
    parser.add_argument('--action-noise', type=float, default=0.0,
                        help='add noise to the actions')
    parser.add_argument('--use-s', action='store_false', default=True,
                        help='learn r(s)')
    parser.add_argument('--use-s-a', action='store_true', default=False,
                        help='learn r(s,a)')
    parser.add_argument('--use-s-a-sprime', action='store_true', default=False,
                        help='learn r(s,a,sprime)')
    parser.add_argument('--reward-epsilon', type=float, default=0.0,
                        help='probability of receiving a random reward')
    parser.add_argument('--reward-high', type=float, default=0.0,
                        help='upper bound on random reward')
    parser.add_argument('--reward-low', type=float, default=0.0,
                        help='lower bound on random reward')
    parser.add_argument('--use-gaussian-noise', action='store_true', default=False,
                        help='use gaussian noise suite')
    parser.add_argument('--use-uniform-noise', action='store_true', default=False,
                        help='use uniform noise suite')
    parser.add_argument('--use-sparse-noise', action='store_true', default=False,
                        help='use sparse noise suite')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    assert not (args.reward_predictor and args.r_s_a)

    return args
