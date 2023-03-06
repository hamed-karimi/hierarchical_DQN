from Agent import Agent
from Environment import Environment
from Controller import Controller
from MetaController import MetaController
import inspect
from copy import deepcopy


class ObjectFactory:
    def __init__(self, utility):
        self.agent = None
        self.environment = None
        self.controller = None
        self.meta_controller = None
        self.params = utility.get_params()

    def get_agent(self, pre_location):
        agent = Agent(self.params.HEIGHT, self.params.WIDTH, n=self.params.NUM_OBJECTS,
                      episode_num=self.params.EPISODE_NUM, episode_len=self.params.EPISODE_LEN,
                      prob_init_needs_equal=self.params.PROB_OF_INIT_NEEDS_EQUAL, predefined_location=pre_location,
                      rho_function=self.params.RHO_FUNCTION, epsilon_function=self.params.EPSILON_FUNCTION)
        self.agent = deepcopy(agent)
        return agent

    def get_environment(self, probability_map, pre_located_objects):  # pre_located_objects is a 2D list
        curr_frame = inspect.currentframe()
        call_frame = inspect.getouterframes(curr_frame, 2)
        if 'meta_controller' in call_frame[1][3]:
            num_objects = self.params.NUM_OBJECTS
        else:
            num_objects = 1
        env = Environment(self.params.HEIGHT, self.params.WIDTH, self.agent, probability_map,
                          reward_of_object=self.params.REWARD_OF_OBJECT,
                          far_objects_prob=self.params.PROB_OF_FAR_OBJECTS_FOR_TWO,
                          num_object=num_objects, pre_located_objects=pre_located_objects,
                          cost_scaler=self.params.COST_SCALER)
        self.environment = deepcopy(env)
        return env

    def get_controller(self):
        controller = Controller(self.params.CONTROLLER_TRAINING_PHASE,
                                self.params.CONTROLLER_BATCH_SIZE,
                                self.params.GAMMA,
                                self.params.EPISODE_NUM,
                                self.params.EPISODE_LEN,
                                self.params.CONTROLLER_MEMORY_CAPACITY,
                                self.params.REWARDED_ACTION_SAMPLING_PROBABILITY_RATIO)
        self.controller = deepcopy(controller)
        return controller

    def get_meta_controller(self):
        meta_controller = MetaController(self.params.META_CONTROLLER_BATCH_SIZE,
                                         self.params.NUM_OBJECTS,
                                         self.params.GAMMA,
                                         self.params.EPISODE_NUM,
                                         self.params.EPISODE_LEN,
                                         self.params.META_CONTROLLER_MEMORY_CAPACITY,
                                         self.params.REWARDED_ACTION_SAMPLING_PROBABILITY_RATIO)
        self.meta_controller = deepcopy(meta_controller)
        return meta_controller

    def get_saved_objects(self):
        return deepcopy(self.agent), deepcopy(self.environment), \
               deepcopy(self.controller), deepcopy(self.meta_controller)
