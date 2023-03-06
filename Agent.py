import numpy as np
import torch
from torch.nn import ReLU
import random
from copy import deepcopy
from itertools import product


class Agent:
    def __init__(self, h, w, n, episode_num, episode_len, prob_init_needs_equal, predefined_location,
                 need_change_different_on_action, rho_function='ReLU',epsilon_function='Linear'):  # n: number of needs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = h
        self.width = w
        self.location = self.initial_location(predefined_location)
        self.num_need = n
        self.range_of_need = [-12, 12]
        self.need_change_on_action = need_change_different_on_action # True of False
        self.prob_init_needs_equal = prob_init_needs_equal
        self.need = self.set_need()
        self.steps_done = 0
        self.episode_num = episode_num
        self.episode_len = episode_len
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.lambda_need = .4  # How much the need increases after each action
        self.relu = ReLU()
        total_need_functions = {'ReLU': self.relu, 'PolyReLU': self.poly_relu}
        self.rho_function = total_need_functions[rho_function]
        self.total_need = self.get_total_need()
        possible_h_w = [list(range(h)), list(range(w))]
        self.epsilon_function = epsilon_function
        self.all_locations = torch.from_numpy(np.array([element for element in product(*possible_h_w)]))

    def poly_relu(self, x, p=2):
        return self.relu(x) ** p

    def set_need(self):
        p = random.uniform(0, 1)
        if p <= self.prob_init_needs_equal:
            need = torch.rand((1, self.num_need))
            need[0, 1:] = need[0, 0]
        else:
            need = torch.rand((1, self.num_need))
        need = (self.range_of_need[1] - self.range_of_need[0]) * need + self.range_of_need[0]
        return need

    def initial_location(self, predefined_location): # predefined_location is a list
        if len(predefined_location[0]) > 0:
            return torch.tensor(predefined_location)
        return torch.from_numpy(np.asarray((np.random.randint(self.height), np.random.randint(self.width)))).unsqueeze(0)

    def update_need_after_step(self, moving_cost):
        for i in range(self.num_need):
            self.need[0, i] += (self.lambda_need + self.need_change_on_action * moving_cost / 4)

    def update_need_after_reward(self, reward):
        self.need = self.need - reward
        for i in range(self.num_need):
            self.need[0, i] = max(self.need[0, i], -12)

    def get_total_need(self):
        total_need = self.rho_function(self.need).sum().squeeze()
        return total_need

    def get_location(self):
        return self.location

    # def reset_location(self, environment):  # Resets the location to somewhere other than the current one
    #     temp = (self.all_locations != self.location.squeeze())
    #     temp = torch.logical_or(temp[:, 0], temp[:, 1])
    #     for obj in range(environment.nObj):
    #         object_nonoccupied = (
    #                     self.all_locations != environment.object_locations[obj, :].squeeze())
    #         object_nonoccupied = torch.logical_or(object_nonoccupied[:, 0], object_nonoccupied[:, 1])
    #         temp = torch.logical_and(temp, object_nonoccupied)
    #     available_locations = [self.all_locations[i, :] for i in range(len(temp)) if temp[i]]
    #     new_location = random.choice(available_locations)
    #     self.location = torch.from_numpy(np.array(new_location)).unsqueeze(0)
    #     environment.update_agent_location_on_map(self)

    def take_action(self, environment, action_id):
        selected_action = environment.allactions[action_id].squeeze()  # to device
        self.location[0] += selected_action
        at_cost = environment.get_cost(action_id)
        self.update_need_after_step(at_cost)

        environment.update_agent_location_on_map(self)
        f, _ = environment.get_reward()
        self.update_need_after_reward(f)
        at_total_need = self.get_total_need()
        last_total_need = self.total_need
        # rho = last_total_need - at_total_need - at_cost
        rho = (-1) * at_total_need - at_cost
        goal_reaching_reward = torch.sub(f, at_cost).squeeze()
        self.total_need = deepcopy(at_total_need)
        return rho.unsqueeze(0), goal_reaching_reward
