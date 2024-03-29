import itertools

from Visualizer import Visualizer
import numpy as np
import torch
import matplotlib.pyplot as plt
from State_batch import State_batch
from copy import deepcopy
from matplotlib.ticker import FormatStrFormatter


def get_predefined_needs(num_object):
    temp_need = [[-10, -5, 0, 5, 10]] * num_object
    need_num = len(temp_need[0]) ** num_object
    need_batch = torch.zeros((need_num, num_object))
    ns = np.zeros((1, num_object))
    for i, ns in enumerate(itertools.product(*temp_need)):
        need_batch[i, :] = torch.tensor(ns)
    return need_batch


class MetaControllerVisualizer(Visualizer):
    def __init__(self, utility):
        super().__init__(utility)
        self.episode_num = utility.params.META_CONTROLLER_EPISODE_NUM
        allactions_np = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]),
                         np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1])]
        self.allactions = [torch.from_numpy(x).unsqueeze(0) for x in allactions_np]
        self.action_mask = np.zeros((self.height, self.width, 1, len(self.allactions)))
        self.initialize_action_masks()
        self.needs = get_predefined_needs(self.object_type_num)
        self.color_options = [[1, 0, .2], [0, .8, .2], [1, 1, 1]]
        self.objects_color_name = ['red', 'green']
        self.row_num = 5
        self.col_num = 6

    def get_figure_title(self, need):
        title = '$n_{0}: {1:.2f}'.format('{'+self.objects_color_name[0]+'}', need[0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{'+self.objects_color_name[i]+'}', need[i])
        return title

    def get_agent_goal_map_from_selected_goal(self, i, j, goal_index, object_locations):
        agent_goal_map = torch.zeros((1, 2, self.height, self.width))
        agent_goal_map[0, 0, i, j] = 1
        if goal_index == object_locations.shape[0]: # stay
            agent_goal_map[0, 1, :, :] = agent_goal_map[0, 0, :, :].clone()
        else: # goal
            agent_goal_map[0, 1, object_locations[goal_index, 0], object_locations[goal_index, 1]] = 1
        return agent_goal_map

    def get_goal_directed_actions(self, object_locations, object_layers, meta_controller, controller):
        num_object = object_locations.shape[0]
        action_values = torch.zeros((self.height, self.width))
        which_action = torch.zeros((self.height, self.width), dtype=torch.int16)
        which_goal = torch.zeros((self.height, self.width), dtype=torch.int16)
        row_num = 5
        col_num = 5
        fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
        for fig_num, need in enumerate(self.needs):
            for i in range(self.height):
                for j in range(self.width):
                    env_map = torch.zeros((1, 1 + num_object, self.height, self.width))  # +1 for agent layer
                    env_map[0, 0, i, j] = 1
                    env_map[0, 1:, :, :] = deepcopy(object_layers)
                    with torch.no_grad():
                        state = State_batch(env_map.to(self.device), need.unsqueeze(0).to(self.device))
                        goals_values = meta_controller.policy_net(state).clone() # 1 * 3
                        goal = goals_values.argmax()
                        which_goal[i, j] = goal
                        agent_goal_map = self.get_agent_goal_map_from_selected_goal(i, j,
                                                                                    goal_index=goal,
                                                                                    object_locations=object_locations).to(self.device)

                        state = State_batch(agent_goal_map, None)
                        controller_values = controller.policy_net(state)
                        action_mask = torch.tensor(self.action_mask[i, j, :, :])
                        controller_values[torch.logical_not(action_mask.bool())] = -3.40e+38
                        action_values[i, j], which_action[i, j] = controller_values.max(1)[0], \
                                                                  controller_values.max(1)[1].detach()
            X = np.arange(0, self.height, 1)
            Y = np.arange(0, self.width, 1)
            arrows_x = np.zeros((self.height, self.width))
            arrows_y = np.zeros((self.height, self.width))
            colors = []
            for i in range(self.height):
                for j in range(self.width):
                    arrows_y[i, j] = -self.allactions[which_action[i, j].int()][0, 0]
                    arrows_x[i, j] = self.allactions[which_action[i, j].int()][0, 1]
                    colors.append(self.color_options[which_goal[i, j]])

            r = fig_num // col_num
            c = fig_num % col_num

            ax[r, c].quiver(X, Y, arrows_x, arrows_y, scale=10, facecolor=colors)

            ax[r, c].set_title(self.get_figure_title(need), fontsize=10)
            # "$n_{{red}}: {:.2f}, n_{{green}}: {:.2f}$".format(need[0], need[1]),

            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()
            for obj in range(object_locations.shape[0]):
                ax[r, c].scatter(object_locations[obj, 1], object_locations[obj, 0], marker='*', s=40,
                                 facecolor=self.color_options[obj])
            ax[r, c].set(adjustable='box')

        plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
        return fig, ax

    def add_needs_plot(self, ax, agent_needs, global_index, r, c):
        ax[r, c].set_prop_cycle('color', self.color_options)
        ax[r, c].plot(agent_needs[:global_index, :], linewidth=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Needs', fontsize=9)
        return ax, r, c + 1

    def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
        ax[r, c].scatter(np.arange(steps_done), kwargs['meta_controller_epsilon'], s=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Meta Controller Epsilon', fontsize=9)
        ax[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # ax[r, c].set_box_aspect(aspect=1)
        return ax, r, c + 1

    def policynet_values(self, object_locations, object_layers, meta_controller):
        num_object = object_locations.shape[0]
        row_num = 5
        col_num = 5
        fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
        for fig_num, need in enumerate(self.needs):
            goals_values_text = []
            for i in range(self.height):
                row_table_texts = []
                for j in range(self.width):
                    env_map = torch.zeros((1, 1 + num_object, self.height, self.width))  # +1 for agent layer
                    env_map[0, 0, i, j] = 1
                    env_map[0, 1:, :, :] = deepcopy(object_layers)
                    with torch.no_grad():
                        state = State_batch(env_map.to(self.device), need.unsqueeze(0).to(self.device))
                        goals_values = meta_controller.policy_net(state).clone()  # 1 * 3
                        row_table_texts.append('\n'.join([str(round(goals_values[0, v].item(), 2)) for v in range(num_object+1)]))
                goals_values_text.append(row_table_texts)

            r = fig_num // col_num
            c = fig_num % col_num

            values_table = ax[r, c].table(cellText=goals_values_text,
                                          rowLabels=np.arange(self.height),
                                          colLabels=np.arange(self.width),
                                          # colWidths=1,
                                          loc='center')
            values_table.auto_set_font_size(False)
            values_table.set_fontsize(8)
            values_table.scale(1, 2)
            ax[r, c].set_title(self.get_figure_title(need), fontsize=10)
            ax[r, c].axis('off')

        plt.tight_layout(pad=0.1, w_pad=1, h_pad=1)
        return fig, ax

    def add_needs_difference_hist(self, ax, agent_needs, needs_range, global_index, r, c):
        ax[r, c].set_prop_cycle('color', self.color_options)
        ax[r, c].hist(agent_needs[:global_index, 0] - agent_needs[:global_index, 1],
                      bins=np.linspace(needs_range[0] - needs_range[1], needs_range[1] - needs_range[0], 49),
                      linewidth=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Needs', fontsize=9)
        return ax, r, c + 1
