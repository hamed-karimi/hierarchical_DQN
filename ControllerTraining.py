import numpy as np
from copy import deepcopy
import torch
from ControllerVisualizer import ControllerVisualizer
import matplotlib.pyplot as plt
import matplotlib as mpl
from ObjectFactory import ObjectFactory
from Utilities import Utilities
from AgentExplorationFunctions import *


def get_existing_controller(utility): # "./tr100000_len100_07-06-2022_11-15/Controller"
    params = utility.params
    factory = ObjectFactory(utility)
    controller = factory.get_controller()
    if params.CONTROLLER_DIRECTORY != '':
        controller.load_target_net_from_memory(params.CONTROLLER_DIRECTORY)

    res_folder = utility.make_res_folder(sub_folder='Controller')
    return deepcopy(controller), res_folder


def training_controller(device):
    utility = Utilities()
    params = utility.params
    if not params.TRAIN_CONTROLLER:
        controller, res_folder = get_existing_controller(utility)
        return controller, res_folder
    res_folder = utility.make_res_folder(sub_folder='Controller')

    global_index = 0
    episodes_mean_controller_reward = 0
    moving_avg_controller_reward = []
    episodes_mean_controller_loss = []
    moving_avg_controller_loss = []

    factory = ObjectFactory(utility)
    controller = factory.get_controller()
    controller_visualizer = ControllerVisualizer(utility)
    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    for episode in range(params.EPISODE_NUM):
        episode_controller_loss = 0
        cum_reward = 0
        action = 0

        agent = factory.get_agent(pre_location=[[], []])
        environment = factory.get_environment(few_many=['few', 'many'],
                                              probability_map=environment_initialization_prob_map,
                                              pre_located_objects=[[], []])
        while action < params.EPISODE_LEN:
            last_goal_map = environment.env_map.clone().to(device)
            action_id = controller.get_action(environment, environment.env_map, episode).clone().to(device)
            _, reward = agent.take_action(environment, action_id)
            internal_reward = reward.unsqueeze(0).clone().to(device)
            done = torch.tensor([1]) if internal_reward > 8.0 else torch.tensor([0])
            at_agent_goal_map = environment.env_map.clone().to(device)

            controller.save_experience(last_goal_map, action_id,
                                       at_agent_goal_map, internal_reward, done.to(device),
                                       torch.from_numpy(environment.get_action_mask()).to(device))
            cum_reward += internal_reward.item()
            controller_at_loss = controller.optimize()
            episode_controller_loss += get_controller_loss(controller_at_loss, device).item()
            action += 1
            global_index += 1
            if internal_reward > 8.0:
                break

        episodes_mean_controller_reward += cum_reward / action
        episodes_mean_controller_loss.append(episode_controller_loss / action)
        if (episode + 1) % params.PRINT_OUTPUT == 0:
            moving_avg_controller_reward.append(episodes_mean_controller_reward / params.PRINT_OUTPUT)
            print('avg internal reward', episodes_mean_controller_reward / params.PRINT_OUTPUT, ' ')
            fig, ax, r, c = controller_visualizer.get_greedy_values_figure(controller)

            # r, c = 6, 5
            ax, r, c = controller_visualizer.get_epsilon_plot(ax, r, c,
                                                              controller.steps_done,
                                                              controller_epsilons=controller.epsilon_list)
            for ax_i in range(c, ax.shape[1]):
                fig.delaxes(ax=ax[r, ax_i])
            # controller_visualizer.get_reward_plot(ax, r, c,
            #                                       controller_reward=moving_avg_controller_reward)

            fig.savefig('{0}/episode_{1}.png'.format(res_folder, episode + 1))
            plt.close()
            episodes_mean_controller_reward = 0

        if (episode + 1) % params.CONTROLLER_TARGET_UPDATE == 0:
            controller.update_target_net()
            print('CONTROLLER TARGET NET UPDATED')

        del agent
        del environment

    return controller, res_folder
