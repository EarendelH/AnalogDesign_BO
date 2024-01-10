import argparse
import os
import yaml
from collections import OrderedDict

from ray.tune.registry import get_trainable_cls, register_env

from util.gen_action_sapce import gen_action_space
from util.gen_obs_space import gen_obs_space_extend, flatten_obs_space
# from util.gen_obs_space import gen_obs_space_simple
from util.gen_param_space import gen_param_space
from util.util_func import create_work_dir
from util.assign_param2netlist import assign_param2netlist
from util.run_spectre_simulation import run_spectre_simulation
from util.cal_reward import cal_reward
from util.generalize_config import generalize_config
from util.update_param import update_parameters
from util.update_obs_space import update_obs_space, flatten_observation
# from util.update_obs_space import update_obs_space_simple
from util.normlization import norm_ideal_spec, norm_sim_spec

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import air, tune
import ray


class RllibAnalogDesignAutoEnv(MultiAgentEnv):
    def __init__(self, generalize=False, path='', sim_output=False):

        # Init values
        self.norm_ideal_specs = None
        self.ideal_specs = None
        self.cur_param = None
        self.step_num = 0
        self.max_step = 250

        # Get absolute path
        self.current_path = os.getcwd()

        # Pass generalization flag
        self.generalize = generalize
        self.ideal_specs_path = os.path.join(self.current_path, path)

        # Pass sim_output flag
        self.sim_output_enable = sim_output

        # Set normalization items
        self.norm_specs_file = "config/norm_specs.yaml"
        self.norm_specs_file = os.path.join(self.current_path, self.norm_specs_file)
        with open(self.norm_specs_file, 'r') as file:
            self.norm_specs = yaml.safe_load(file)

        # Set root directory
        self.run_root_dir = "run_test"
        self.run_root_dir = os.path.join(self.current_path, self.run_root_dir)
        if not os.path.exists(self.run_root_dir):
            raise ValueError(f"Root directory {self.run_root_dir} not found.")

        # Load config files
        self.agent_assign_config = "config/agent_assign.yaml"
        self.agent_assign_config = os.path.join(self.current_path, self.agent_assign_config)
        self.result_config = "config/result.yaml"
        self.result_config = os.path.join(self.current_path, self.result_config)
        self.param_range_config = "config/param_range.yaml"
        self.param_range_config = os.path.join(self.current_path, self.param_range_config)
        self.sim_config = "config/simulation.yaml"
        self.sim_config = os.path.join(self.current_path, self.sim_config)
        self.init_param = "config/init_param.yaml"
        self.init_param = os.path.join(self.current_path, self.init_param)

        # Set netlist directory
        self.unassigned_netlist_dir = "netlist_template"
        self.unassigned_netlist_dir = os.path.join(self.current_path, self.unassigned_netlist_dir)

        # Multi-agent config
        with open(self.agent_assign_config, 'r') as file:
            agent_assign = yaml.safe_load(file)

        # RLlib config
        self.possible_agents = list(agent_assign.keys())
        self.agents = self.possible_agents
        self._agent_ids = set(self.agents)

        # Generate param space
        self.param_space = gen_param_space(self.param_range_config)

        # RLlib config
        self.terminateds = set()
        self.truncateds = set()
        self._obs_space_in_preferred_format = True
        self.observation_space = flatten_obs_space(gen_obs_space_extend(self.result_config, self.param_range_config,
                                                                        self.agent_assign_config))
        # self.observation_space = gen_obs_space_simple(self.result_config, self.param_range_config,
        #                                               self.agent_assign_config)
        # print(f"observation_space: {self.observation_space}")
        self._action_space_in_preferred_format = True
        self.action_space = gen_action_space(self.agent_assign_config)
        # print(f"action_space: {self.action_space}")

        self.resetted = False

        super().__init__()

    def reset(self, *, seed=None, options=None):

        # Set the ideal specs based on the generalize flag
        self.ideal_specs = generalize_config(self.generalize, self.ideal_specs_path)
        print(f"Initialing!!!Generalize flag: {self.generalize}")
        print(f"Initialing!!!Ideal specs: {self.ideal_specs}")

        # If self.init_param file exists, use the init_param file as the initial param or
        # Select the middle point of the param space as the initial param

        init_param = OrderedDict()

        if os.path.exists(self.init_param):
            with open(self.init_param, 'r') as file:
                init_param = yaml.safe_load(file)
            print(f"Initialing!!!init_param file exist, Init param: {init_param}")
        else:
            for param, value_list in self.param_space.items():
                n = len(value_list)
                middle_index = n // 2 - 1 if n % 2 == 0 else n // 2
                init_param[param] = value_list[middle_index]
            print(f"Initialing!!!init_param file not exist, Init param: {init_param}")

        # Generate working directory
        working_dir = create_work_dir(self.run_root_dir)
        print(f"Initialing!!!Working directory: {working_dir}")

        for unassigned_netlist_file in os.listdir(self.unassigned_netlist_dir):
            if unassigned_netlist_file.endswith(".scs"):
                assigned_netlist_file = unassigned_netlist_file.replace("_parameterized", "")

                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(init_param, unassigned_netlist_file_path, assigned_netlist_file_path,
                                     assigned_yaml_file_path)

        # Run spectre simulation
        with open(self.sim_config, 'r') as file:
            sim_config = yaml.safe_load(file)

        sim_result = run_spectre_simulation(working_dir, sim_config, self.sim_output_enable)

        # Normalize the current ideal specs
        print(f"Initialing!!!Ideal specs: {self.ideal_specs}")
        self.norm_ideal_specs = norm_ideal_spec(self.ideal_specs, self.norm_specs)
        print(f"Initialing!!!Normalized ideal specs: {self.norm_ideal_specs}")

        # Normalize the current simulation specs
        print(f"Initialing!!!Simulation result: {sim_result}")
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
        print(f"Initialing!!!Normalized simulation result: {norm_sim_result}")

        # Generate observation
        # observation = update_obs_space_simple(self.ideal_specs, norm_sim_result, init_param)
        observation_detail = update_obs_space(self.norm_ideal_specs, norm_sim_result, init_param)
        print(f"Initialing!!!Observation result: {observation_detail}")
        observation = flatten_observation(observation_detail)

        # Share all observations among agents
        observations = {agent: observation for agent in self.agents}
        # print(f"Initialing!!!Observations result: {observations}")

        # Test Rew func
        rew = cal_reward(self.ideal_specs, sim_result)
        print(f"Debug!!!Initialing!!!Reward result: {rew}")

        self.cur_param = init_param

        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()

        info = {agent: {} for agent in self.agents}

        return observations, info

    def step(self, action_dict):

        # Update step number
        self.step_num += 1

        # print(f"Updated action: {action_dict} with step number: {self.step_num}")

        # Create working directory
        working_dir = create_work_dir(self.run_root_dir)
        print(f"Step!!! Working directory: {working_dir} with step number: {self.step_num}")

        print(f"Step!!!Actions: {action_dict} with step number: {self.step_num}")
        # Flatten all actions
        all_action_flatten = OrderedDict()
        for group in action_dict.values():
            for key, value in group.items():
                all_action_flatten[key] = value
        # print(f"Step!!!Flatten actions: {all_action_flatten}")

        # Update param with new action
        # print(f"Step!!!Current param index: {self.cur_param} with step number: {self.step_num}")
        print(f"Step!!!Previous param: {self.cur_param} with step number: {self.step_num}")
        updated_param = update_parameters(all_action_flatten, self.cur_param, self.param_range_config)
        print(f"Step!!!Updated param: {updated_param} with step number: {self.step_num}")

        # Update current param
        self.cur_param = updated_param

        # Parse the updated param and generate the netlist
        for unassigned_netlist_file in os.listdir(self.unassigned_netlist_dir):
            if unassigned_netlist_file.endswith(".scs"):
                assigned_netlist_file = unassigned_netlist_file.replace("_parameterized", "")

                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(updated_param, unassigned_netlist_file_path, assigned_netlist_file_path,
                                     assigned_yaml_file_path)

        # Run spectre simulation
        with open(self.sim_config, 'r') as file:
            sim_config = yaml.safe_load(file)

        # Run spectre simulation and normalize the result
        sim_result = run_spectre_simulation(working_dir, sim_config, self.sim_output_enable)
        print(f"Step!!!Simulation result: {sim_result} with step number: {self.step_num}")
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
        # print(f"Step!!!Normalized simulation result: {norm_sim_result} with step number: {self.step_num}")

        # Generate observation
        # observation = update_obs_space_simple(self.ideal_specs, norm_sim_result, updated_param)
        observation_detail = update_obs_space(self.norm_ideal_specs, norm_sim_result, updated_param)
        print(f"Step!!!Observation result: {observation_detail} with step number: {self.step_num}")
        observation = flatten_observation(observation_detail)

        # Share all observations
        observations = {agent: observation for agent in self.agents}
        # print(f"Step!!!Observations result: {observations} with step number: {self.step_num}")

        # Store the observation
        with open(os.path.join(working_dir, "result.yaml"), 'w') as file:
            yaml.dump(observation, file)

        print("Step!!!self.ideal_specs: ", self.ideal_specs)
        # print("Step!!!observation_detail: ", observation_detail)

        # Calculate reward
        rew = {a: -10 for a in self.agents}
        rew_single = cal_reward(self.ideal_specs, sim_result)
        for agent_name in rew:
            rew[agent_name] = rew_single
        print(f"Step!!!Reward result: {rew} with step number: {self.step_num}")

        # Determine termination or truncations
        terminated = {a: False for a in self.agents}
        for agent_name in terminated:
            if rew[agent_name] >= 0:
                terminated[agent_name] = True

        truncated = {a: False for a in self.agents}
        for agent_name in truncated:
            if self.step_num >= self.max_step:
                truncated[agent_name] = True

        info = {agent: {} for agent in self.agents}

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        print(f"Step!!!terminated: {terminated} with step number: {self.step_num}")
        print(f"Step!!!truncated: {truncated} with step number: {self.step_num}")

        return observations, rew, terminated, truncated, info
