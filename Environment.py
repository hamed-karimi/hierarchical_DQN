import random
import torch
import numpy as np
import warnings


class Environment:
    def __init__(self, few_many_objects, h, w, agent, probability_map, reward_of_object, far_objects_prob, num_object, pre_located_objects):  # object_type_num is the number of each type of object
        self.each_type_object_num = None
        self.channels = 1 + num_object  # +1 is for agent which is the first layer
        self.height = h
        self.width = w
        self.object_type_num = num_object
        self.few_many_objects = few_many_objects
        self.probability_map = probability_map
        self.far_objects_prob = far_objects_prob
        self.agent_location = agent.get_location()
        self.env_map = torch.zeros((1, self.channels, self.height, self.width),
                                   dtype=torch.float32)  # the 1 is for the env_map can be matched with the dimesions of weights (8, 2, 4, 4)
        self.object_locations = self.init_object_locations(pre_located_objects)
        self.update_agent_location_on_map(agent)
        self.reward_of_object = [reward_of_object] * agent.num_need
        self.cost_of_staying = 1 # this should change for meta controller
        allactions_np = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]),
                         np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1])]
        self.allactions = [torch.from_numpy(x).unsqueeze(0) for x in allactions_np]
        self.check_obj_need_compatability(agent.num_need)

    def check_obj_need_compatability(self, num_agent_need):
        if self.object_type_num != num_agent_need:
            warnings.warn("The number of needs and objects are not equal")

    def get_each_object_type_appearance(self):
        # e.g., self.few_many_objects : ['few', 'many']
        few_range = np.array([1, 2])
        many_range = np.array([3, 4, 5])
        nums = {'few': np.random.choice(few_range),
                'many': np.random.choice(many_range)}
        max_num = max(nums['few'], nums['many'])
        each_type_object_num = []
        for item in self.few_many_objects:
            each_type_object_num.append(nums[item])
        object_locations = -1 * torch.ones(self.object_type_num, max_num, 2, dtype=torch.int32)
        return each_type_object_num, object_locations

    def init_objects_randomly(self, pre_located_objects):  # pre_located_objects is a list
        # object_locations = torch.zeros((self.object_type_num, 2), dtype=torch.int16)
        # for i in range(len(pre_located_objects)):
        #     if len(pre_located_objects[i]) > 0:
        #         object_locations[i, :] = torch.Tensor(pre_located_objects[i])
        #         self.env_map[0, 1 + i, pre_located_objects[i][0], pre_located_objects[i][1]] = 1
        if self.object_type_num == 1:  # for controller
            object_locations = -1 * torch.ones(self.object_type_num, 1, 2, dtype=torch.int32)
            self.each_type_object_num = [1, 0]
        else:
            self.each_type_object_num, object_locations = self.get_each_object_type_appearance()

        for obj_type in range(self.object_type_num):
            for at_obj in range(self.each_type_object_num[obj_type]):
                if len(pre_located_objects[obj_type]) > 0:
                    continue
                do = 1
                while do:
                    # sample = np.array([np.random.randint(self.height), np.random.randint(self.width)])
                    hw_range = np.arange(self.height * self.width)
                    rand_num_in_range = random.choices(hw_range, weights=self.probability_map, k=1)[0]
                    sample = torch.tensor([rand_num_in_range//self.width, rand_num_in_range%self.width])
                    if sum(self.env_map[0, 1:, sample[0], sample[1]]) == 0:  # This location is empty on every object layer
                        object_locations[obj_type, at_obj, :] = sample
                        self.env_map[0, 1+obj_type, sample[0], sample[1]] = 1
                        # self.probability_map[rand_num_in_range] *= .9
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

    def init_object_locations(self, pre_located_objects):  # Place objects on the map
        p = random.uniform(0, 1)
        if p <= self.far_objects_prob:
            return self.init_two_objects_far_from_each_other()
        else:
            return self.init_objects_randomly(pre_located_objects)

    def update_agent_location_on_map(self, agent):
        # This is called by the agent (take_action method) after the action is taken
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
        r = torch.zeros((1, self.object_type_num), dtype=torch.float32)
        goal_reached = torch.zeros(self.object_type_num, dtype=torch.float32)
        for obj in range(self.object_type_num):
            goal_reached[obj] = torch.all(torch.eq(self.agent_location[0], self.object_locations[obj, :, :]),
                                          dim=1).any().item()

            r[0, obj] += (goal_reached[obj] * self.reward_of_object[obj])
        return r, goal_reached

    def get_cost(self, action_id):
        if action_id == 0:
            return self.cost_of_staying
        return torch.linalg.norm(self.allactions[action_id].squeeze().float())
