# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states

    def predict_reward(self, s, a, s_prime):
        r_hat_input = self.format_r_input(s, a, s_prime)
        return self.rp_forward(r_hat_input)


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, num_heads=1, hidden_size=512):
        super(CNNPolicy, self).__init__()
        self.num_heads = num_heads

        self.representations = []
        self.critics = []
        for _ in range(num_heads):
            self.representations.append(self.build_representation(num_inputs, hidden_size=hidden_size))
            self.critics.append(self.build_critic(hidden_size, 1))

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(hidden_size, num_outputs)
        else:
            raise NotImplementedError

        self.critics = nn.ModuleList(self.critics)
        self.representations = nn.ModuleList(self.representations)

        self.param_groups = [list(self.parameters())]

        self.train()
        self.reset_parameters()

    def build_representation(self, num_inputs, hidden_size=512):
        return nn.ModuleList([nn.Conv2d(num_inputs, 32, 8, stride=4),
                              nn.Conv2d(32, 64, 4, stride=2),
                              nn.Conv2d(64, 32, 3, stride=1),
                              nn.Linear(32 * 7 * 7, hidden_size)])

    def build_critic(self, num_inputs, num_outputs):
        return nn.ModuleList([nn.Linear(num_inputs, num_outputs)])

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        for rep in self.representations:
            for l in rep:
                l.weight.data.mul_(relu_gain)

        for critic in self.critics:
            for i in range(len(critic) - 1):
                critic[i].weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        representations = []
        for rep in self.representations:
            x = rep[0](inputs / 255.0)
            x = F.relu(x)

            x = rep[1](x)
            x = F.relu(x)

            x = rep[2](x)
            x = F.relu(x)

            x = x.view(-1, 32 * 7 * 7)

            if len(rep) == 4:
                x = rep[3](x)
                x = F.relu(x)
            representations.append(x)

        x = representations[-1]

        value = []
        for c, critic in enumerate(self.critics):
            cur_rep = representations[0] if c > len(representations) - 1 else representations[c]
            for i in range(len(critic)):
                if i == len(critic) - 1:
                    value.append(critic[i](cur_rep))
                else:
                    cur_rep = critic[i](cur_rep)
                    cur_rep = F.relu(cur_rep)

        value = [value] if type(value) is not list else value
        value = torch.cat(value, dim=-1)

        return value, x, states


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, num_heads=1,
                 reward_predictor=False, use_s=True, use_s_a=False, use_s_a_sprime=False):
        assert use_s + use_s_a + use_s_a_sprime <= 1
        super(MLPPolicy, self).__init__()

        self.use_s = use_s
        self.use_s_a = use_s_a
        self.use_s_a_sprime = use_s_a_sprime

        self.num_heads = num_heads

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.critics = []
        self.param_groups = [list(self.parameters())]

        cur_critic = self.build_critic(num_inputs, num_outputs=num_heads, hidden_size=64)
        self.critics.append(cur_critic)

        self.critics = nn.ModuleList(self.critics)

        for critic in list(self.critics):
            self.param_groups.append(list(critic.parameters()))

        if reward_predictor:
            if self.use_s:
                r_hat_input_size = num_inputs
            elif self.use_s_a:
                r_hat_input_size = num_inputs + num_outputs
            else:
                r_hat_input_size = num_inputs * 2 + num_outputs

            self.rp = self.build_critic(r_hat_input_size, num_outputs=1, hidden_size=64)
            self.param_groups.append(list(self.rp.parameters()))
        self.train()
        self.reset_parameters()

    def build_critic(self, num_inputs, num_outputs, hidden_size=None):
            return nn.ModuleList([nn.Linear(num_inputs, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, num_outputs)])

    def format_r_input(self, s, a, s_prime):
        if self.use_s:
            return s
        elif self.use_s_a:
            return torch.cat([s, a], dim=-1)
        else:  # use s_a_s'
            return torch.cat([s, a, s_prime], dim=-1)


    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        value = []
        for c, critic in enumerate(self.critics):
            cur_rep = inputs
            for i in range(len(critic)):
                if i == len(critic) - 1:
                    value.append(critic[i](cur_rep))
                else:
                    cur_rep = critic[i](cur_rep)
                    cur_rep = F.tanh(cur_rep)

        value = [value] if type(value) is not list else value
        value = torch.cat(value, dim=-1)

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x, states

    def rp_forward(self, inputs):
        x = inputs
        for i, l in enumerate(self.rp):
            x = l(x)
            if i != len(self.rp) - 1:
                x = F.tanh(x)
        return x
