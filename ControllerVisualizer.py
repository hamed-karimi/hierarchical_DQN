from Visualizer import Visualizer
import numpy as np
import torch
import matplotlib.pyplot as plt
from State_batch import State_batch
from matplotlib.ticker import FormatStrFormatter


class ControllerVisualizer(Visualizer):
    def __init__(self, utility):
        super().__init__(utility)

        self.row_num = 3
        self.col_num = 6

    def get_greedy_values_figure(self, controller):
        fig, ax = plt.subplots(self.row_num, self.col_num, figsize=(15, 10))
        for x in range(self.height):
            for y in range(self.width):
                action_values = torch.zeros((self.height, self.width))
                which_action = torch.zeros((self.height, self.width), dtype=torch.int16)
                for i in range(self.height):
                    for j in range(self.width):
                        env_map = torch.zeros((1, 2, self.height, self.width))
                        env_map[0, 0, i, j] = 1
                        env_map[0, 1, x, y] = 1
                        with torch.no_grad():
                            state = State_batch(env_map.to(self.device), None)
                            controller_values = controller.policy_net(state)
                            action_mask = torch.tensor(self.action_mask[i, j, :, :])
                            controller_values[torch.logical_not(action_mask.bool())] = -3.40e+38
                            action_values[i, j], which_action[i, j] = controller_values.max(1)[0], \
                                                                      controller_values.max(1)[1].detach()
                Xs = np.arange(0, self.height, 1)
                Ys = np.arange(0, self.width, 1)
                arrows_x = np.zeros((self.height, self.width))
                arrows_y = np.zeros((self.height, self.width))
                for i in range(self.height):
                    for j in range(self.width):
                        arrows_y[i, j] = -self.allactions[which_action[i, j].int()][0, 0]
                        arrows_x[i, j] = self.allactions[which_action[i, j].int()][0, 1]

                fig_num = x*self.height + y
                r = fig_num // self.col_num
                c = fig_num % self.col_num
                ax[r, c].quiver(Xs, Ys, arrows_x, arrows_y, scale=10)
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].invert_yaxis()
                ax[r, c].scatter(y, x, marker='*', s=40, facecolor=[1, 0, .2])
                ax[r, c].set_box_aspect(aspect=1)

        plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
        return fig, ax

    def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
        ax[r, c].scatter(np.arange(steps_done), kwargs['controller_epsilons'], s=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Controller Epsilon', fontsize=9)
        ax[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[r, c].set_box_aspect(aspect=1)
        return ax, r, c + 1

