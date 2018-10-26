# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torch.nn as nn
import numpy as np


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


class ReplayBuffer(object):
    def __init__(self, size, sample_rewards_evenly=False):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.sample_rewards_evenly = sample_rewards_evenly
        self._storage = []
        self._reward_indices = set()
        self._non_reward_indices = set()
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, true_obs_t):
        data = (obs_t, action, reward, obs_tp1, done, true_obs_t)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)

        else:
            self._storage[self._next_idx] = data

        if reward > 0.5 or reward < -0.5:
            if self._next_idx not in self._reward_indices:
                self._reward_indices.add(self._next_idx)
            if self._next_idx in self._non_reward_indices:
                self._non_reward_indices.remove(self._next_idx)
        else:
            if self._next_idx not in self._non_reward_indices:
                self._non_reward_indices.add(self._next_idx)
            if self._next_idx in self._reward_indices:
                self._reward_indices.remove(self._next_idx)

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, true_obses_t = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, true_obs_t = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            true_obses_t.append(np.array(true_obs_t, copy=False))

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), \
               np.array(true_obses_t)

    def sample(self, batch_size, split_batch=False, just_biased=False):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = []
        if not just_biased:
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        if self.sample_rewards_evenly and (split_batch or just_biased):
            idxes_r = random.sample(self._reward_indices, int(batch_size / 2))
            idxes_nr = random.sample(self._non_reward_indices, int(batch_size / 2))
            idxes += idxes_r + idxes_nr

        return self._encode_sample(idxes)

    def sample_batched(self, batch_size):
        idxes = np.arange(self.__len__())
        np.random.shuffle(idxes)
        for i in range(0, self.__len__(), batch_size):
            end_index = min(i + batch_size, self.__len__())
            yield self._encode_sample(idxes[i:end_index])