import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


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
        super(hDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.fc1 = nn.Linear(32 + 2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3) # 0, 1: goals, 2: stay

    def forward(self, state_batch):
        env_map = state_batch.env_map
        agent_need = state_batch.agent_need
        y = F.relu(self.conv1(env_map))
        y = y.view(-1, num_flat_features(y))
        y = torch.cat((y, agent_need), 1)  # Adding the needs

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class lDQN(nn.Module):  # controller network
    def __init__(self):
        super(lDQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 4)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 9)

    def forward(self, state_batch):
        x = state_batch.env_map.clone() # agent map and goal map (the second layer contains a single 1)
        y = F.relu(self.conv1(x))
        y = y.view(-1, num_flat_features(y))
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y
