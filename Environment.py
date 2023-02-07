import random

import torch
import numpy as np
import warnings


class Environment:
    def __init__(self, h, w, agent, probability_map, reward_of_object, far_objects_prob, num_object):  # nObj is the number of each type of object
        self.channels = 1 + num_object  # +1 is for agent which is the first layer
        self.height = h
        self.width = w
        self.nObj = num_object
        self.probability_map = probability_map
        self.far_objects_prob = far_objects_prob
        self.agent_location = agent.get_location()
        self.env_map = torch.zeros((1, self.channels, self.height, self.width),
                                   dtype=torch.float32)  # the 1 is for the env_map can be matched with the dimesions of weights (8, 2, 4, 4)
        self.object_locations = self.init_object_locations()
        self.update_agent_location_on_map(agent)
        self.reward_of_object = [reward_of_object] * agent.num_need
        self.cost_of_staying = 0
        allactions_np = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]),
                         np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1])]
        self.allactions = [torch.from_numpy(x).unsqueeze(0) for x in allactions_np]
        self.check_obj_need_compatibality(agent.num_need)

    def check_obj_need_compatibality(self, num_agent_need):
        if self.nObj != num_agent_need:
            warnings.warn("The number of needs and objects are not equal")

    def init_objects_randomly(self):
        object_locations = torch.zeros((self.nObj, 2), dtype=torch.int16)
        for at_obj in range(self.nObj):
            do = 1
            while do:
                # sample = np.array([np.random.randint(self.height), np.random.randint(self.width)])
                hw_range = np.arange(self.height * self.width)
                rand_num_in_range = random.choices(hw_range, weights=self.probability_map, k=1)[0]
                sample = np.array([rand_num_in_range//self.width, rand_num_in_range%self.width])
                if sum(self.env_map[0, 1:, sample[0], sample[1]]) == 0:  # This location is empty on every object layer
                    object_locations[at_obj, :] = torch.from_numpy(sample)
                    self.env_map[0, 1+at_obj, sample[0], sample[1]] = 1
                    self.probability_map[rand_num_in_range] *= .9
                    do = 0
        return object_locations

    def init_two_objects_far_from_each_other(self):
        r = random.random()
        if r < .5:
            row1 = random.choice([0, self.height - 1])
            col1 = random.randint(0, self.width - 1)
            while True:
                row2 = self.height - 1 if row1 == 0 else 0
                col2 = random.randint(0, self.width - 1)
                if abs(row2 - row1) + abs(col2 - col1) >= self.height:
                    break
        else:
            row1 = random.randint(0, self.height - 1)
            col1 = random.choice([0, self.width - 1])
            while True:
                row2 = random.randint(0, self.height - 1)
                col2 = self.width - 1 if col1 == 0 else 0
                if abs(row2 - row1) + abs(col2 - col1) >= self.width:
                    break
        self.env_map[0, 1, row1, col1] = 1
        self.env_map[0, 2, row2, col2] = 1
        object_locations = torch.tensor([[row1, col1], [row2, col2]])

        return object_locations

    def init_object_locations(self):  # Place objects on the map
        p = random.uniform(0, 1)
        if p <= self.far_objects_prob:
            return self.init_two_objects_far_from_each_other()
        else:
            return self.init_objects_randomly()

    def update_agent_location_on_map(self,
                                     agent):  # This is called by the agent (take_action method) after the action is taken
        self.env_map[0, 0, self.agent_location[0, 0], self.agent_location[0, 1]] = 0
        self.agent_location = agent.get_location().clone()
        self.env_map[0, 0, self.agent_location[0, 0], self.agent_location[0, 1]] = 1

    def get_all_action(self):
        return self.allactions

    def get_action_mask(self):
        aa = np.ones((self.agent_location.size(0), len(self.allactions)))
        for i, location in enumerate(self.agent_location):
            if location[0] == 0:
                aa[i, 2] = 0
                aa[i, 6] = 0
                aa[i, 7] = 0
            if location[0] == self.height - 1:
                aa[i, 1] = 0
                aa[i, 5] = 0
                aa[i, 8] = 0
            if location[1] == 0:
                aa[i, 4] = 0
                aa[i, 6] = 0
                aa[i, 8] = 0
            if location[1] == self.width - 1:
                aa[i, 3] = 0
                aa[i, 5] = 0
                aa[i, 7] = 0
        return aa

    def get_reward(self):
        r = torch.zeros((1, self.nObj), dtype=torch.float32)
        goal_reached = torch.zeros((self.nObj), dtype=torch.float32)
        for obj in range(self.nObj):
            goal_reached[obj] = torch.all(self.agent_location[0] == self.object_locations[obj, :]).int()
            r[0, obj] += (goal_reached[obj] * self.reward_of_object[obj])
        return r, goal_reached

    def get_cost(self, action_id):
        if action_id == 0:
            return self.cost_of_staying
        return torch.linalg.norm(self.allactions[action_id].squeeze().float())
