import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

import Utilities


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def num_flat_features(x):  # This is a function we added for convenience to find out the number of features in a layer.
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class hDQN(nn.Module):  # meta controller network
    def __init__(self):
        utilities = Utilities.Utilities()
        params = utilities.params
        super(hDQN, self).__init__()
        env_layer_num = params.OBJECT_TYPE_NUM + 1  # +1 for agent layer
        padding = (8 - params.WIDTH) // 2
        kernel_size = min(params.WIDTH, params.DQN_CONV1_KERNEL)
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=kernel_size,
                               padding=padding)
        self.max_pool = nn.MaxPool2d(params.WIDTH-kernel_size+1)
        self.fc1 = nn.Linear(in_features=params.DQN_CONV1_OUT_CHANNEL + params.OBJECT_TYPE_NUM,
                             out_features=16)  # needs are equal to # of objects
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, env_layer_num)  # 0, 1: goals, 2: stay

    def forward(self, state_batch):
        env_map = state_batch.env_map
        agent_need = state_batch.agent_need
        y = F.relu(self.conv1(env_map))
        y = F.relu(self.max_pool(y))
        y = y.flatten(start_dim=1, end_dim=-1)
        y = torch.cat((y, agent_need), 1)  # Adding the needs
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class lDQN(nn.Module):  # controller network
    def __init__(self):
        super(lDQN, self).__init__()
        utilities = Utilities.Utilities()
        params = utilities.params
        env_layer_num = params.OBJECT_TYPE_NUM  # +1 for agent layer
        # padding = (8 - params.WIDTH) // 2
        kernel_size = min(params.WIDTH, params.DQN_CONV1_KERNEL)
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=kernel_size)
        self.max_pool = nn.MaxPool2d(params.WIDTH-kernel_size+1)
        self.fc1 = nn.Linear(in_features=params.DQN_CONV1_OUT_CHANNEL + params.OBJECT_TYPE_NUM,
                             out_features=16)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 9)

    def forward(self, state_batch):
        x = state_batch.env_map.clone() # agent map and goal map (the second layer contains a single 1)
        y = F.relu(self.conv1(x))
        y = F.relu(self.max_pool(y))
        y = y.flatten(start_dim=1, end_dim=-1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y
