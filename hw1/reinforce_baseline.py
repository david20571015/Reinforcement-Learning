# Spring 2022, IOC 5259 Reinforcement Learning
# HW1-partII: REINFORCE and baseline

from itertools import count

import gym
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and
          the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s),
               the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """

    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[
            0]
        self.hidden_size = 128

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.policy_network = nn.Linear(self.hidden_size, self.action_dim)
        self.value_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1))
        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.action_log_probs = []
        self.values = []
        self.rewards = []

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        x = self.shared_layer(state)
        action_logits = self.policy_network(x)
        state_value = self.value_network(x)
        ########## END OF YOUR CODE ##########

        return action_logits, state_value

    def select_action(self, state: torch.Tensor) -> float:
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        action_logits, state_value = self(state)
        distrib = Categorical(logits=action_logits)
        action = distrib.sample()
        state_value = torch.squeeze(state_value)
        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.action_log_probs.append(distrib.log_prob(action))
        self.values.append(state_value)

        return action.item()

    def calculate_loss(self, gamma: float = 0.99) -> torch.Tensor:
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        discounted_sum = 0
        for reward in reversed(self.rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()

        returns = torch.Tensor(returns)
        std, mean = torch.std_mean(returns)
        returns = (returns - mean) / std

        action_log_probs = torch.stack(self.action_log_probs, dim=0)
        values = torch.stack(self.values, dim=0)

        # Do not comptue gradient of the value network
        # because it is used to predict the return.
        # with torch.no_grad():
        advantage = returns - values
        policy_loss = -(advantage * action_log_probs).sum()

        value_loss = F.mse_loss(values, returns, reduction='sum')
        ########## END OF YOUR CODE ##########

        return policy_loss + value_loss

    def clear_memory(self) -> None:
        # reset rewards and action buffer
        del self.action_log_probs[:]
        del self.values[:]
        del self.rewards[:]


def train(lr: float = 0.01, gamma: float = 0.99) -> None:
    """
        Train the model using SGD (via backpropagation)
        TODO: In each episode,
            1. run the policy till the end of the episode and keep the sampled trajectory
            2. update both the policy and the value network at the end of episode
    """

    # Instantiate the policy model and the optimizer
    model = Policy()
    # writer.add_graph(model, torch.zeros(1, model.observation_dim))
    # return
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler

        # For each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        ########## YOUR CODE HERE (10-15 lines) ##########

        while t < 10000:
            t += 1
            state = torch.Tensor(state).view(1, -1)
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)

            ep_reward += reward
            model.rewards.append(reward)

            if done:
                break

        optimizer.zero_grad()
        loss = model.calculate_loss(gamma=gamma)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.clear_memory()
        ########## END OF YOUR CODE ##########
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\t'
              f'reward: {ep_reward}\tewma reward: {ewma_reward}\tloss: {loss.item()}')

        writer.add_scalar('reward/ep_reward', ep_reward, i_episode)
        writer.add_scalar('reward/ewma_reward', ewma_reward, i_episode)

        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold or i_episode % 2000 == 0:
            torch.save(model.state_dict(),
                       f'./pre_trained/{task}_lr_{lr}_gamma_{gamma}.pth')
            print(f'Solved! Running reward is now {ewma_reward} and'
                  f'the last episode runs to {t} time steps!')
            break


def test(name: str, n_episodes: int = 10) -> None:
    """
        Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(torch.load(f'./pre_trained/{name}'))

    render = False
    max_episode_len = 10000

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len + 1):
            state = torch.Tensor(state).view(1, -1)
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f'Episode {i_episode}\tReward: {running_reward}')
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 42
    task = 'LunarLander-v2'
    lr = 0.01
    gamma = 0.99
    env = gym.make(task)
    env.reset(seed=random_seed)
    torch.manual_seed(random_seed)
    writer = SummaryWriter(f'logs/{task}/lr_{lr}_gamma_{gamma}')
    train(lr, gamma)
    test(f'{task}_lr_{lr}_gamma_{gamma}.pth')
