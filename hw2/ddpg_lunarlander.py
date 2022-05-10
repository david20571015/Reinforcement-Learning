# Spring 2022, IOC 5259 Reinforcement Learning
# HW2: DDPG

import os
import random
import time
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Transition = namedtuple('Transition',
                        ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise(object):

    def __init__(self,
                 action_dimension,
                 scale=0.1,
                 mu=0,
                 theta=0.15,
                 sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Sigmoid(),
        )
        ########## END OF YOUR CODE ##########

    def forward(self, inputs):
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = self.net(inputs)
        x = x * 2.0 - 1.0  # Linear mapping to [-1.0, 1.0]
        return x
        ########## END OF YOUR CODE ##########


class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        self.critic_net = nn.Sequential(nn.Linear(num_inputs, hidden_size))
        self.action_net = nn.Sequential(nn.Linear(num_outputs, hidden_size))
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        x = self.critic_net(inputs)
        y = self.action_net(actions)
        return self.net(x + y)
        ########## END OF YOUR CODE ##########


class DDPG(object):

    def __init__(self,
                 num_inputs,
                 action_space,
                 gamma=0.995,
                 tau=0.0005,
                 hidden_size=128,
                 lr_a=1e-4,
                 lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs,
                                  self.action_space)
        # self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs,
                                    self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(Variable(state))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed
        noise = [0.0] if action_noise is None else action_noise.noise()
        noise = torch.FloatTensor(noise)
        return torch.clamp(mu + noise, min=-1.0, max=1.0)
        ########## END OF YOUR CODE ##########

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat([trans.state for trans in batch]))
        action_batch = Variable(torch.cat([trans.action for trans in batch]))
        reward_batch = Variable(torch.cat([trans.reward for trans in batch
                                          ])).view(-1, 1)
        mask_batch = Variable(torch.cat([trans.mask for trans in batch
                                        ])).view(-1, 1)
        next_state_batch = Variable(
            torch.cat([trans.next_state for trans in batch]))

        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic
        with torch.no_grad():
            y_batch = reward_batch + self.gamma * mask_batch * self.critic_target(
                next_state_batch, self.actor_target(next_state_batch))

        value_loss = F.mse_loss(
            y_batch,
            self.critic(state_batch, action_batch),
        )
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        policy_loss = -self.critic(
            state_batch,
            self.actor(state_batch),
        ).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        ########## END OF YOUR CODE ##########

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self,
                   env_name,
                   suffix='',
                   actor_path=None,
                   critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime('%m%d%Y_%H%M%S', local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = f'preTrained/ddpg_actor_{env_name}_{timestamp}_{suffix}'
        if critic_path is None:
            critic_path = f'preTrained/ddpg_critic_{env_name}_{timestamp}_{suffix}'
        print(f'Saving models to {actor_path} and {critic_path}')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print(f'Loading models from {actor_path} and {critic_path}')
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


def train():
    num_episodes = 2000
    gamma = 0.995
    tau = 0.0002
    hidden_size = 256
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 128
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    updates = 0
    lr_a = 1e-4
    lr_c = 1e-3

    agent = DDPG(
        env.observation_space.shape[0],
        env.action_space,
        gamma,
        tau,
        hidden_size,
        lr_a,
        lr_c,
    )
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    for i_episode in range(num_episodes):

        ounoise.scale = noise_scale
        ounoise.reset()

        state = torch.Tensor(env.reset()).view(1, -1)

        episode_reward = 0

        total_numsteps = 0
        while True:
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            action = agent.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(
                action.cpu().detach().numpy()[0])
            total_numsteps += 1

            memory.push(
                state,
                action,
                torch.Tensor([0.0 if done and total_numsteps <= 999 else 1.0]),
                torch.Tensor(next_state).view(1, -1),
                torch.Tensor([reward]),
            )

            if total_numsteps % updates_per_step == 0 and batch_size <= len(
                    memory):
                batch = memory.sample(batch_size)
                agent.update_parameters(batch)

            episode_reward += reward
            state = torch.Tensor(next_state).view(1, -1)

            if done:
                break
            ########## END OF YOUR CODE ##########

        rewards.append(episode_reward)

        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor(env.reset()).view(1, -1)
            episode_reward = 0
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(
                    action.cpu().detach().numpy()[0])

                # env.render()

                episode_reward += reward
                state = torch.Tensor(next_state).view(1, -1)

                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)
            print(f'Episode: {i_episode},\tlength: {t},\t'
                  f'reward: {rewards[-1]:.2f},\tewma reward: {ewma_reward:.2f}')

            writer.add_scalar('episode_reward/reward', episode_reward,
                              i_episode)
            writer.add_scalar('ewma_reward/reward', ewma_reward, i_episode)

    agent.save_model(ENV_NAME, '.pth')


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    RANDOM_SEED = 10
    ENV_NAME = 'LunarLanderContinuous-v2'
    env = gym.make(ENV_NAME)
    env.reset(seed=RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    writer = SummaryWriter(f'logs/{ENV_NAME}')
    train()
    writer.close()
