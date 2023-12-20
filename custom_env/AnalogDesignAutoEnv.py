import functools
import os
import yaml
from collections import OrderedDict
from copy import copy

from util.gen_action_sapce import gen_action_space
from util.gen_obs_space import gen_obs_space
from util.gen_param_space import gen_param_space
from util.util_func import create_work_dir
from util.assign_param2netlist import assign_param2netlist
from util.run_spectre_simulation import run_spectre_simulation
from util.parse_action_idx import parse_action_idx
from util.cal_reward import cal_reward
from util.generalize_config import generalize_config

from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType


class AnalogDesignEnv(ParallelEnv):
    metadata = {
        "name": "Analog_Design_Env_v0",
    }

    def __init__(self, generalize=False, path=''):

        # Init values
        self.cur_param = None
        self.step_num = 0
        self.max_step = 10000

        # Set the ideal specs based on the generalize flag
        self.ideal_specs = generalize_config(generalize, path)

        # Set root directory
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = "run_test"
        self.root_dir_path = os.path.join(file_path, self.root_dir)
        if not os.path.exists(self.root_dir_path):
            raise ValueError(f"Root directory {self.root_dir_path} not found.")

        # Load config files
        self.agent_assign_config = "config/agent_assign.yaml"
        self.result_config = "config/result.yaml"
        self.param_range_config = "config/param_range.yaml"
        self.sim_config = "config/simulation.yaml"

        # Set netlist directory
        self.unassigned_netlist_dir = "netlist_template"

        # Multi-agent config
        with open(self.agent_assign_config, 'r') as file:
            agent_assign = yaml.safe_load(file)

        self.possible_agents = list(agent_assign.keys())

        # Generate param space
        self.param_space = gen_param_space(self.param_range_config)

    def reset(self, seed=None, options=None):

        self.agents = copy(self.possible_agents)

        # Select the middle point of the param space as the initial param
        init_param = OrderedDict()
        for param, value_list in self.param_space.items():
            n = len(value_list)
            middle_index = n // 2 - 1 if n % 2 == 0 else n // 2
            init_param[param] = value_list[middle_index]
        print(f"Initialing!!!Init param: {init_param}")

        # Generate working directory
        working_dir = create_work_dir(self.root_dir)
        print(f"Initialing!!!Working directory: {working_dir}")

        # Parse the init param and generate the netlist
        for unassigned_netlist_file in os.listdir(self.unassigned_netlist_dir):
            if unassigned_netlist_file.endswith(".scs"):
                assigned_netlist_file = unassigned_netlist_file.replace("_parameterized", "")

                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(unassigned_netlist_file_path, assigned_netlist_file_path, assigned_yaml_file_path,
                                     init_param)

        # Run spectre simulation
        with open(self.sim_config, 'r') as file:
            sim_config = yaml.safe_load(file)

        observations = run_spectre_simulation(working_dir, sim_config)
        print(f"Initialing!!!Simulation result: {observations}")

        # Inherit data
        self.cur_param = init_param

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):

        # Update step number
        self.step_num += 1

        # Create working directory
        working_dir = create_work_dir(self.root_dir)
        print(f"Step!!! Working directory: {working_dir}")

        # Summary all actions
        all_action = {}
        for agent_name, single_agent_action in actions:
            all_action[agent_name] = single_agent_action
        # Flatten all actions
        all_action_flatten = OrderedDict()
        for action in all_action.values():
            all_action_flatten.update(action)

        print(f"New Action: {all_action_flatten} with step number: {self.step_num}")

        # Update param with new action
        updated_action_idx = OrderedDict()
        for key in all_action_flatten:
            updated_action_idx[key] = all_action_flatten[key] + self.cur_param[key]
        print(f"Updated param index: {updated_action_idx} with step number: {self.step_num}")

        # Parse updated param index -> param
        updated_param = parse_action_idx(self.param_space, updated_action_idx)
        print(f"Updated param: {updated_param} with step number: {self.step_num}")

        # Parse the updated param and generate the netlist
        for unassigned_netlist_file in os.listdir(self.unassigned_netlist_dir):
            if unassigned_netlist_file.endswith(".scs"):
                assigned_netlist_file = unassigned_netlist_file.replace("_parameterized", "")

                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(unassigned_netlist_file_path, assigned_netlist_file_path, assigned_yaml_file_path,
                                     updated_param)

        # Run spectre simulation
        with open(self.sim_config, 'r') as file:
            sim_config = yaml.safe_load(file)

        # Share all observations
        observations = run_spectre_simulation(working_dir, sim_config)
        print(f"Step!!!Simulation result: {observations} with step number: {self.step_num}")

        # Calculate reward
        rewards = {a: -10 for a in self.agents}
        for agent_name in rewards:
            rewards[agent_name] = cal_reward(observations, self.ideal_specs)

        # Determine termination or truncations
        terminations = {a: False for a in self.agents}
        for agent_name in terminations:
            if rewards[agent_name] > 0:
                terminations[agent_name] = True

        truncations = {a: False for a in self.agents}
        for agent_name in truncations:
            if self.step_num >= self.max_step:
                truncations[agent_name] = True

        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observation_space = gen_obs_space(self.result_config, self.param_range_config)
        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        all_action_space = gen_action_space(self.agent_assign_config)
        action_space = all_action_space[agent]
        return action_space
