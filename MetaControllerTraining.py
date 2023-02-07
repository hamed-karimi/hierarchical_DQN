import numpy as np
from copy import deepcopy
import torch
from MetaControllerVisualizer import MetaControllerVisualizer
from Visualizer import get_reward_plot, get_loss_plot
from ObjectFactory import ObjectFactory
from Utilities import Utilities
from AgentExplorationFunctions import *


def training_meta_controller(controller):
    utility = Utilities('Parameters.json')
    params = utility.get_params()

    res_folder = utility.make_res_folder(sub_folder='MetaController')

    global_index = 0
    meta_controller_reward_list = []
    meta_controller_reward_sum = 0
    meta_controller_loss_list = []
    num_goal_selected = [0, 0, 0] # 0, 1: goals, 2: stay
    agent_needs_over_time = np.zeros((params.EPISODE_NUM * params.EPISODE_LEN, 2), dtype=np.float16)

    factory = ObjectFactory(utility)
    meta_controller = factory.get_meta_controller()
    meta_controller_visualizer = MetaControllerVisualizer(utility)
    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    for episode in range(params.EPISODE_NUM):
        episode_meta_controller_reward = 0
        episode_meta_controller_loss = 0
        action = 0
        agent = factory.get_agent(need_num=2)
        environment = factory.get_environment(environment_initialization_prob_map, num_object=2)
        initial_env_map = environment.env_map.clone()
        initial_need = agent.need.clone()
        goal_map, goal_index = meta_controller.get_goal_map(environment, agent, episode)
        num_goal_selected[goal_index] += 1
        goal_reached = False
        done = torch.tensor([0])
        while True:
            last_agent_goal_map = torch.stack(
                [environment.env_map[0, 0, :, :], goal_map], dim=0) \
                .unsqueeze(0).clone()
            action_id = controller.get_action(environment, last_agent_goal_map, episode).clone()
            rho, _ = agent.take_action(environment, action_id)
            at_loss = meta_controller.optimize()
            episode_meta_controller_loss = get_meta_controller_loss(at_loss)

            goal_reached = agent_reached_goal(agent, environment, goal_index)
            episode_meta_controller_reward += rho
            agent_needs_over_time[global_index, :] = agent.need.clone()
            action += 1
            global_index += 1

            if goal_reached:
                done = torch.tensor([1])
            if goal_reached or action < params.EPISODE_LEN or rho >= 0:
                break
        meta_controller.save_experience(initial_env_map, initial_need, goal_index,
                                        episode_meta_controller_reward, done,
                                        environment.env_map.clone(),
                                        agent.need.clone())
        meta_controller_reward_sum += episode_meta_controller_reward.item()
        meta_controller_loss_list.append((episode_meta_controller_loss / action))
        if (episode + 1) % params.PRINT_OUTPUT == 0:
            meta_controller_reward_list.append(meta_controller_reward_sum/params.PRINT_OUTPUT)
            print('avg meta controller reward', meta_controller_reward_sum/params.PRINT_OUTPUT)
            meta_controller_reward_sum = 0
            fig, ax = meta_controller_visualizer.get_goal_directed_actions(environment.object_locations.clone(),
                                                                           environment.env_map[0, 1:, :, :],
                                                                           meta_controller, controller)
            fig.savefig('{0}/episode_{1}.png'.format(res_folder, episode + 1))
            plt.close()
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            r, c = 0, 0

            ax, r, c = get_reward_plot(ax, r, c,
                                       reward=meta_controller_reward_list,
                                       title="Meta Controller Reward")

            ax, r, c = get_loss_plot(ax, r, c, loss=meta_controller_loss_list,
                                     title='Meta Controller Loss')

            r, c = 1, 0
            ax, r, c = meta_controller_visualizer.get_epsilon_plot(ax, r, c, meta_controller.steps_done,
                                                                   meta_controller_epsilon=meta_controller.epsilon_list)

            # meta_controller_visualizer.add_selected_goals_plot(ax, r, c, episode,
            #                                                                meta_controller.selected_goal)
            # meta_controller_visualizer.add_needs_plot(ax, agent_needs_over_time, global_index, r, c)

            meta_controller_visualizer.add_needs_difference_hist(ax, agent_needs_over_time, agent.range_of_need, global_index, r, c)
            fig.savefig('{0}/training_proc_episode_{1}.png'.format(res_folder, episode + 1))
            plt.close()

        if (episode + 1) % params.META_CONTROLLER_TARGET_UPDATE == 0:
            meta_controller.update_target_net()
            print('META CONTROLLER TARGET NET UPDATED')
    return meta_controller, res_folder
