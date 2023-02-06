import os.path
from datetime import datetime

import torch

from AgentExplorationFunctions import *
from TrainingController import training_controller
from MetaControllerTraining import training_meta_controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = datetime.now()
controller, controller_dir = training_controller(device)

print("Controller trained in: ", datetime.now() - start_time)
print('Saving Controller...')

controller_model_path = os.path.join(controller_dir, 'controller_model.pt')
torch.save(controller.target_net.state_dict(), controller_model_path)

meta_controller, meta_controller_dir = training_meta_controller(controller)
meta_controller_model_path = os.path.join(meta_controller_dir, 'meta_controller_model.pt')
torch.save(meta_controller.target_net.state_dict(), meta_controller_model_path)
