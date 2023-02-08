import torch
import matplotlib.pyplot as plt


def agent_reached_goal(agent, environment, goal_index):
    target_goal_layer = 0 if goal_index == environment.nObj else goal_index+1
    agent_object_maps_equality = torch.all(torch.eq(environment.env_map[0, 0, :, :],
                                                    environment.env_map[0, target_goal_layer, :, :]))
    if agent_object_maps_equality:
        return True
    return False
    # if torch.all(torch.eq(agent.location, environment.object_locations[goal_index, :])):
    #     return True
    # return False


def get_controller_loss(controller_at_loss, device):
    if torch.is_tensor(controller_at_loss):
        return controller_at_loss
    else:
        return torch.Tensor([0.0]).to(device)


def get_meta_controller_loss(meta_controller_at_loss):
    if torch.is_tensor(meta_controller_at_loss):
        return meta_controller_at_loss.detach().item()
    else:
        return 0.0
