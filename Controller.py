import os.path
import random
from torch import optim
import torch
from State_batch import State_batch
from DQN import lDQN, weights_init_orthogonal
from ReplayMemory import ReplayMemory
from collections import namedtuple
from torch import nn
import numpy as np
import math


class ControllerMemory(ReplayMemory):
    def __init__(self, capacity):
        super().__init__(capacity=capacity)

    def get_transition(self, *args):
        Transition = namedtuple('Transition',
                                ('agent_goal_map', 'action',
                                 'next_agent_goal_map',
                                 'reward', 'done', 'action_mask'))
        return Transition(*args)


class Controller:

    def __init__(self, training_phase, batch_size, gamma, episode_num, episode_len, memory_capacity, rewarded_action_selection_ratio):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_phase = training_phase
        self.policy_net = lDQN().to(self.device)
        self.policy_net.apply(weights_init_orthogonal)
        self.target_net = lDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ControllerMemory(capacity=memory_capacity)
        self.rewarded_action_selection_ratio = rewarded_action_selection_ratio
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.batch_size_mul = 3
        self.steps_done = 0
        self.epsilon_list = []

    def load_target_net_from_memory(self, path):
        model_path = torch.load(os.path.join(path, 'controller_model.pt'))
        self.policy_net.load_state_dict(model_path)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_non_linear_epsilon(self, step_done):
        whole = self.episode_num * self.episode_len * .6
        steep = math.log(whole)  # 3*end/4990 + 992/499
        k = 1 / (whole / steep)
        i = whole / 2
        f_x = np.exp(k * (step_done - i))
        epsilon = 1 / (1 + f_x)
        return epsilon

    def get_linear_epsilon(self, episode):
        epsilon = self.EPS_START - (episode / self.episode_num) * \
                                    (self.EPS_START - self.EPS_END)
        return epsilon

    def get_action(self, environment, goal_map, episode):
        epsilon = self.get_linear_epsilon(episode)
        self.epsilon_list.append(epsilon)
        e = random.random()
        action_mask = environment.get_action_mask()
        if e < epsilon and self.training_phase:
            valid_actions = np.where(action_mask > 0)[1]  ### The agent is taking some invalid actions
            action_id = np.array(valid_actions[random.randint(0, valid_actions.shape[0] - 1)])
            action_id = torch.from_numpy(action_id).unsqueeze(0)
        else:
            with torch.no_grad():
                env_map = goal_map.clone()
                state = State_batch(env_map.to(self.device), None)
                action_values = self.policy_net(state).squeeze()
                action_values[torch.logical_not(torch.from_numpy(action_mask).bool())[0]] = -3.40e+38
                action_id = action_values.argmax().unsqueeze(0)
        self.steps_done += 1
        return action_id

    def save_experience(self, initial_agent_goal_map, action, next_agent_goal_map, reward, done, action_mask):
        self.memory.push_experience(initial_agent_goal_map, action, next_agent_goal_map, reward, done, action_mask)
        if done.item() == 1:
            self.memory.push_selection_ratio(selection_raio=self.rewarded_action_selection_ratio)
        else:
            self.memory.push_selection_ratio(selection_raio=1)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        if self.memory.__len__() < self.BATCH_SIZE * self.batch_size_mul:
            return float('nan')
        transition_sample = self.memory.sample(self.BATCH_SIZE)
        batch = self.memory.get_transition(*zip(*transition_sample))

        initial_agent_goal_map_batch = torch.cat([batch.agent_goal_map[i] for i in range(len(batch.agent_goal_map))])
        action_batch = torch.cat(batch.action)
        at_agent_goal_map_batch = torch.cat(
            [batch.next_agent_goal_map[i] for i in range(len(batch.next_agent_goal_map))])
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        mask_batch = torch.cat(batch.action_mask)

        at_state_batch = State_batch(at_agent_goal_map_batch, None)
        initial_state_batch = State_batch(initial_agent_goal_map_batch, None)

        policynet_goal_values_of_initial_state = self.policy_net(initial_state_batch)
        targetnet_goal_values_of_at_state = self.target_net(at_state_batch)
        targetnet_goal_values_of_at_state[torch.logical_not(mask_batch.bool())] = -3.40e+38

        targetnet_max_goal_value = targetnet_goal_values_of_at_state.max(1)[0].detach().float()
        goal_values_of_selected_goals = policynet_goal_values_of_initial_state \
            .gather(dim=1, index=action_batch.unsqueeze(1))
        expected_goal_values = (1 - done_batch) * targetnet_max_goal_value * self.GAMMA + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(goal_values_of_selected_goals, expected_goal_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
