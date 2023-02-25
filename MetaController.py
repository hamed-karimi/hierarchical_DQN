import random
from torch import optim
import torch
from State_batch import State_batch
from DQN import hDQN, weights_init_orthogonal
from ReplayMemory import ReplayMemory
from collections import namedtuple
from torch import nn
import numpy as np
import math


class MetaControllerMemory(ReplayMemory):
    def __init__(self, capacity):
        super().__init__(capacity=capacity)

    def get_transition(self, *args):
        Transition = namedtuple('Transition',
                                ('initial_map', 'initial_need', 'goal_index', 'cum_reward', 'done', 'final_map',
                                 'final_need'))
        return Transition(*args)


class MetaController:

    def __init__(self, batch_size, gamma, episode_num, episode_len, memory_capacity, rewarded_action_selection_ratio):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = hDQN().to(self.device)
        self.policy_net.apply(weights_init_orthogonal)
        self.target_net = hDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = MetaControllerMemory(memory_capacity)
        self.rewarded_action_selection_ratio = rewarded_action_selection_ratio
        self.steps_done = 0
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.batch_size_mul = 1
        self.epsilon_list = []
        # self.selected_goal = np.ones((self.episode_num, 2)) * -1

    def get_nonlinear_epsilon(self, episode):
        x = math.log(episode + 1, self.episode_num)
        epsilon = -x ** 40 + 1
        return epsilon

    def get_linear_epsilon(self, episode):
        epsilon = self.EPS_START - (episode / self.episode_num) * \
                  (self.EPS_START - self.EPS_END)
        return epsilon

    def get_goal_map(self, environment, agent, episode):
        # epsilon = self.get_nonlinear_epsilon(episode)
        epsilon = self.get_linear_epsilon(episode)
        self.epsilon_list.append(epsilon)
        object_locations = environment.object_locations.clone()
        e = random.random()
        if e < epsilon:  # random (goal or stay)
            ind = random.randint(0, object_locations.shape[0])
            ind = torch.tensor(ind).clone().detach()
        else:
            with torch.no_grad():
                env_map = environment.env_map.clone().to(self.device)
                need = agent.need.to(self.device)
                state = State_batch(env_map, need)
                goal_values = self.policy_net(state).squeeze()
                ind = goal_values.argmax().cpu()

        # stay
        if ind == object_locations.shape[0]:
            self.steps_done += 1
            goal_map = environment.env_map[0, 0, :, :].clone()  # agent map as goal map
            return goal_map, ind.unsqueeze(0)

        # goal
        selected_obj_location = object_locations[ind, :]
        goal_map = torch.zeros((environment.height, environment.width), dtype=torch.float32)
        goal_map[selected_obj_location[0], selected_obj_location[1]] = 1
        self.steps_done += 1
        # self.selected_goal[episode, ind] = 1 if episode == 0 else self.selected_goal[episode-1, ind]+1
        # self.selected_goal[episode, 1-ind] = 0 if episode == 0 else self.selected_goal[episode-1, 1-ind]
        return goal_map, ind.unsqueeze(0)

    def save_experience(self, initial_map, initial_need, goal_index, acquired_reward, done, final_map, final_need):
        self.memory.push_experience(initial_map, initial_need, goal_index, acquired_reward, done, final_map, final_need)
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        memory_prob = relu(acquired_reward) + 1  # This should be changed to sigmoid
        self.memory.push_selection_ratio(selection_ratio=memory_prob)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        if self.memory.__len__() < self.BATCH_SIZE * self.batch_size_mul:
            return float('nan')
        transition_sample = self.memory.sample(self.BATCH_SIZE)
        batch = self.memory.get_transition(*zip(*transition_sample))

        initial_map_batch = torch.cat([batch.initial_map[i] for i in range(len(batch.initial_map))]).to(self.device)
        initial_need_batch = torch.cat([batch.initial_need[i] for i in range(len(batch.initial_need))]).to(self.device)
        goal_indices_batch = torch.cat(batch.goal_index).to(self.device)
        cum_reward_batch = torch.cat(batch.cum_reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        final_map_batch = torch.cat([batch.final_map[i] for i in range(len(batch.final_map))]).to(self.device)
        final_need_batch = torch.cat([batch.final_need[i] for i in range(len(batch.final_need))]).to(self.device)

        final_state_batch = State_batch(final_map_batch, final_need_batch)
        initial_state_batch = State_batch(initial_map_batch, initial_need_batch)

        policynet_goal_values_of_initial_state = self.policy_net(initial_state_batch).to(self.device)
        targetnet_goal_values_of_final_state = self.target_net(final_state_batch).to(self.device)

        targetnet_max_goal_value = targetnet_goal_values_of_final_state.max(1)[0].detach().float()
        goal_values_of_selected_goals = policynet_goal_values_of_initial_state \
            .gather(dim=1, index=goal_indices_batch.unsqueeze(1))
        expected_goal_values = (1 - done_batch) * targetnet_max_goal_value * self.GAMMA + cum_reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(goal_values_of_selected_goals, expected_goal_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
