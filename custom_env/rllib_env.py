import os
import yaml
from collections import OrderedDict
import pickle
import datetime
import random
import subprocess
import logging

from util.gen_action_space import gen_masked_action_space
from util.gen_obs_space import gen_obs_space_w_region, flatten_obs_space_w_region
from util.gen_param_space import gen_param_space
from util.util_func import create_work_dir
from util.assign_param2netlist import update_netlist
from util.run_spectre_simulation import run_dynamic_simulation, run_region_simulation
from util.cal_reward import cal_reward
from util.generalize_config import generalize_config
from util.update_param import update_parameters
from util.update_obs_space import update_obs_space_w_region, flatten_observation_w_region
from util.normlization import norm_ideal_spec, norm_sim_spec
from util.util_func import retry_decorator
from util.init_param import gen_init_param
from util.extract_device_param_value import extract_operation_region_w_name

from ray.rllib.env.multi_agent_env import MultiAgentEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RllibAnalogDesignAutoEnv(MultiAgentEnv):
    def __init__(self, generalize=True,
                 netlist_folder_name='',
                 specs_folder_name='',
                 config_folder_name='',
                 run_folder_name='',
                 sim_output=False,
                 init_method='file',
                 action_mask=True,
                 dc_check=True):

        # Init values
        self.resetted = None
        self.trajectory_data = None
        self.log_file_path = None
        self.log_file_name = None
        self.zero_sim_result = None
        self.step_num = None
        self.norm_ideal_specs = None
        self.ideal_specs = None
        self.cur_param = None
        self.max_step = 1024

        # Get absolute path
        self.current_path = os.getcwd()

        # Pass action mask flag
        self.action_mask = action_mask

        # Pass initial dc check flag
        self.dc_check = dc_check

        # Pass generalization flag
        self.generalize = generalize
        self.ideal_specs_path = os.path.join(self.current_path, specs_folder_name)

        # Pass init method
        self.init_method = init_method

        # Pass sim_output flag
        self.sim_output_enable = sim_output

        # Get home directory
        self.home_dir = os.path.expanduser("~")

        # Set root directory based on home directory
        # Avoid run folder placing in /tmp directory when running on cluster
        # ~/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/ + self.run_root_dir
        self.run_root_dir = run_folder_name
        self.run_root_dir = os.path.join(self.home_dir, "AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env",
                                         self.run_root_dir)
        if not os.path.exists(self.run_root_dir):
            raise ValueError(f"Root directory {self.run_root_dir} not found.")

        # Load config files
        self.agent_assign_config = config_folder_name + "/agent_assign.yaml"
        self.agent_assign_config = os.path.join(self.current_path, self.agent_assign_config)
        self.param_range_config = config_folder_name + "/param_range.yaml"
        self.param_range_config = os.path.join(self.current_path, self.param_range_config)
        self.sim_config = config_folder_name + "/simulation.yaml"
        self.sim_config = os.path.join(self.current_path, self.sim_config)
        self.dc_sim_config = config_folder_name + "/simulation_region.yaml"
        self.dc_sim_config = os.path.join(self.current_path, self.dc_sim_config)
        self.predefined_init_param = config_folder_name + "/init_param.yaml"
        self.predefined_init_param = os.path.join(self.current_path, self.predefined_init_param)
        self.device_mask_config = config_folder_name + "/device_mask.yaml"
        self.device_mask_config = os.path.join(self.current_path, self.device_mask_config)
        self.norm_specs_file = config_folder_name + "/norm_specs.yaml"
        self.norm_specs_file = os.path.join(self.current_path, self.norm_specs_file)

        # Set netlist directory
        self.unassigned_netlist_dir = netlist_folder_name
        self.unassigned_netlist_dir = os.path.join(self.current_path, self.unassigned_netlist_dir)

        # Load YAML
        with open(self.sim_config, 'r') as file:
            self.sim_config_dict = yaml.safe_load(file)
        # Generate DC sim config
        with open(self.norm_specs_file, 'r') as file:
            self.norm_specs = yaml.safe_load(file)
        with open(self.param_range_config, 'r') as file:
            self.param_range_dict = yaml.safe_load(file)
        with open(self.agent_assign_config, 'r') as file:
            agent_assign_dict = yaml.safe_load(file)
        with open(self.device_mask_config, 'r') as file:
            self.device_mask_dict = yaml.safe_load(file)
        with open(self.dc_sim_config, 'r') as file:
            self.dc_sim_config_dict = yaml.safe_load(file)

        # RLlib config
        self.possible_agents = list(agent_assign_dict.keys())
        self.agents = self.possible_agents
        self._agent_ids = set(self.agents)

        # Generate param space
        self.param_space = gen_param_space(self.param_range_dict)
        logging.debug(f"param_space: {self.param_space}")

        # Create an empty dict for operation region
        self.operation_region_dict_zero = {}
        for component, data in self.param_range_dict.items():
            if component == 'other_variable':
                pass
            else:
                self.operation_region_dict_zero[component] = 0

        # RLlib config
        self.terminateds = set()
        self.truncateds = set()
        self._obs_space_in_preferred_format = True

        # Generate observation and action space
        self.observation_space = flatten_obs_space_w_region(gen_obs_space_w_region(self.sim_config_dict,
                                                                                   self.param_range_dict,
                                                                                   agent_assign_dict))
        logging.debug(f"observation_space: {self.observation_space}")
        self._action_space_in_preferred_format = True
        self.action_space = gen_masked_action_space(action_mask, self.device_mask_dict, agent_assign_dict)
        logging.debug(f"action_space: {self.action_space}")

        self.resetted = False

        super().__init__()

    def reset(self, *, seed=None, options=None):
        # Reset step number
        self.step_num = 0

        # Set the ideal specs based on the generalize flag
        self.ideal_specs = generalize_config(self.generalize, self.ideal_specs_path)
        logging.info(f"Initialing!!!Generalize flag: {self.generalize}")
        logging.info(f"Initialing!!!Ideal specs: {self.ideal_specs}")

        init_param = gen_init_param(self.init_method, self.predefined_init_param, self.action_mask,
                                    self.device_mask_dict, self.param_space)
        logging.debug(f"Initialing!!!Init param: {init_param}")

        # Generate working directory
        working_dir_reset = create_work_dir(self.run_root_dir)
        logging.info(f"Initialing!!!Working directory: {working_dir_reset}")

        # Update Netlist File
        update_netlist(working_dir_reset, self.sim_config_dict, init_param, self.unassigned_netlist_dir)

        # For avoid simulation error in step, generate a default result with zero value but correct key in step method
        self.zero_sim_result = {}
        for sim in self.sim_config_dict:
            sim_name = sim['simulation_name']
            sim_items = sim['simulation_item']
            self.zero_sim_result[sim_name] = {item: 0.0 for item in sim_items}
            if sim_name == 'DC':
                self.zero_sim_result[sim_name]['IQ'] = 1.0
            if sim_name.startswith('Trans'):
                self.zero_sim_result[sim_name] = {item: 1.0 for item in sim_items}
            # Add Simulation Name before each keys in the result dictionary. Avoid error in flatten the dictionary
            modified_result = {f"{sim_name}_{key}": value for key, value in self.zero_sim_result[sim_name].items()}
            self.zero_sim_result[sim_name] = modified_result
        logging.info(f"Initialing!!!Zero sim result: {self.zero_sim_result}")

        try:
            sim_result, _ = run_dynamic_simulation(working_dir_reset, self.sim_config_dict, self.zero_sim_result,
                                                   self.sim_output_enable)
        # For avoid simulation error in init, use zero result instead.
        except Exception as e:
            logging.warning(f"Warning!!!: {e}. Simulation failed, use zero result instead.")
            sim_result = self.zero_sim_result

        # Normalize the current ideal specs
        logging.info(f"Initialing!!!Ideal specs: {self.ideal_specs}")
        self.norm_ideal_specs = norm_ideal_spec(self.ideal_specs, self.norm_specs)
        logging.debug(f"Initialing!!!Normalized ideal specs: {self.norm_ideal_specs}")
        logging.debug(f"Initialing!!!Ideal specs: {self.ideal_specs}")

        # Normalize the current simulation specs
        logging.info(f"Initialing!!!Simulation result: {sim_result}")
        logging.debug(f"Initialing!!!Ideal specs: {self.ideal_specs}")
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
        logging.debug(f"Initialing!!!Normalized simulation result: {norm_sim_result}")

        # Generate region observation
        try:
            # Run psf for converting binary file to text file
            working_dir_reset_dc = create_work_dir(self.run_root_dir)
            logging.info(f"Initialing!!!DC Check working directory: {working_dir_reset_dc}")
            update_netlist(working_dir_reset_dc, self.dc_sim_config_dict, init_param, self.unassigned_netlist_dir)
            run_region_simulation(working_dir_reset_dc, self.dc_sim_config_dict, self.zero_sim_result, self.sim_output_enable)
            dc_reset_raw_result_path = os.path.join(working_dir_reset_dc, "Region.raw/dcOpInfo.info")
            dc_reset_result_path = os.path.join(working_dir_reset_dc, "Region.raw/dcOpInfo.info.encode")
            subprocess.run(f"psf {dc_reset_raw_result_path} -o {dc_reset_result_path}", shell=True)
            reset_operation_region_dict = extract_operation_region_w_name(dc_reset_result_path)
            logging.debug(f"Initialing!!!Operation region: {reset_operation_region_dict}")
            # delete_work_dir(working_dir_reset_dc)
        except Exception as e:
            logging.warning(f"Resting!!!: {e}. No DC sim file.")
            reset_operation_region_dict = self.operation_region_dict_zero

        # Check operation_region_dict length vs self.operation_region_dict_zero length
        if len(reset_operation_region_dict) != len(self.operation_region_dict_zero):
            logging.warning(f"Resting!!!: Operation region dict length {len(reset_operation_region_dict)} "
                            f"does not match with operation_region_dict_zero length {len(self.norm_ideal_specs)}.")
            reset_operation_region_dict = self.operation_region_dict_zero

        # Generate observation
        observation_detail = update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result, init_param,
                                                       reset_operation_region_dict)
        logging.debug(f"Initialing!!!Observation detail: {observation_detail}")
        observation = flatten_observation_w_region(observation_detail)
        logging.debug(f"Initialing!!!Flatten Observation: {observation}")

        # Share all observations among agents
        observations = {agent: observation for agent in self.agents}

        # Test Rew func
        rew = cal_reward(self.ideal_specs, sim_result, self.norm_specs)
        logging.info(f"Debug!!!Initialing!!!Reward result: {rew}")

        self.cur_param = init_param

        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()

        info = {agent: {} for agent in self.agents}

        # Generate log pickle file
        self.log_file_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(1000, 9999)}.pkl"
        self.log_file_path = os.path.join(self.run_root_dir, self.log_file_name)

        # Save trajectory reset info
        self.trajectory_data = {
            'initial_data': {
                'ideal_specs': self.ideal_specs,
                'norm_specs': self.norm_specs,
                'sim_result': sim_result,
                'init_param': init_param,
                'rew': rew
            },
            'steps_data': []
        }

        with open(self.log_file_path, 'wb') as f:
            pickle.dump(self.trajectory_data, f)

        # Delete working temp directory, if it exists
        # delete_work_dir(working_dir_reset)

        return observations, info

    def step(self, action_dict):

        # Update step number
        self.step_num += 1

        operation_region_dict = {}

        if self.action_mask:
            for key in self.device_mask_dict:
                for agent_key in action_dict:
                    if key in action_dict[agent_key]:
                        new_values = self.device_mask_dict[key]
                        for new_key in new_values:
                            new_value = action_dict[agent_key][key]
                            action_dict[agent_key][new_key] = new_value

        # Flatten all actions
        all_action_flatten = OrderedDict()
        for group in action_dict.values():
            for key, value in group.items():
                all_action_flatten[key] = value

        # Update param with new action
        logging.debug(f"Step!!!All action flatten: {all_action_flatten}")
        logging.debug(f"Step!!!Current param: {self.cur_param}")
        logging.debug(f"Step!!!Param range config: {self.param_range_config}")
        updated_param = update_parameters(all_action_flatten, self.cur_param, self.param_range_config)
        logging.info(f"Step!!!Updated param: {updated_param} with step number: {self.step_num}")

        # Update current param
        self.cur_param = updated_param

        # Run DC check firstly and only once. If dc_check is True. If DC check failed, return zero sim result and -25
        # reward. End the episode.
        valid_param = None
        if self.dc_check:
            try:
                working_dir_step_dc = create_work_dir(self.run_root_dir)
                logging.info(f"Step!!!DC Check Working directory: {working_dir_step_dc} "
                             f"with step number: {self.step_num}")
                # Update DC Netlist File for checking operation region
                update_netlist(working_dir_step_dc, self.dc_sim_config_dict, updated_param, self.unassigned_netlist_dir)
                run_region_simulation(working_dir_step_dc, self.dc_sim_config_dict, self.zero_sim_result,
                                      self.sim_output_enable)
                dc_raw_result_path = os.path.join(working_dir_step_dc, "Region.raw/dcOpInfo.info")
                dc_result_path = os.path.join(working_dir_step_dc, "Region.raw/dcOpInfo.info.encode")
                subprocess.run(f"psf {dc_raw_result_path} -o {dc_result_path}", shell=True)
                operation_region_dict = extract_operation_region_w_name(dc_result_path)
                operation_region_list = list(operation_region_dict.values())
                logging.info(f"Step!!!Operation region: {operation_region_list} with step number: {self.step_num}")
                # 0 cut-off, 1 triode, 2 saturation, 3 sub-th, 4 breakdown
                # Check whether all transistors are in saturation/sub-threshold/triode region
                valid_param = all(item in [1, 2, 3] for item in operation_region_list)
                # Check operation_region_dict length vs self.operation_region_dict_zero length
                if len(operation_region_dict) != len(self.operation_region_dict_zero):
                    logging.warning(f"Warning!!!: Operation region dict length {len(operation_region_dict)} does not "
                                    f"match with operation_region_dict_zero length {len(self.norm_ideal_specs)}.")
                    operation_region_dict = self.operation_region_dict_zero
                    valid_param = False
                # Delete working temp directory, if it exists
                # delete_work_dir(working_dir_step_dc)
            except Exception as e:
                logging.warning(f"Warning!!!: {e}. Failed to run DC check with step number: {self.step_num}")
                operation_region_dict = self.operation_region_dict_zero
                valid_param = False

        # DC check fail condition
        if self.dc_check and not valid_param:
            logging.info(f"Step!!!Param failed DC check with step number: {self.step_num}")
            sim_result = self.zero_sim_result
            logging.debug(f"Debug, sim_result is {sim_result}")
            logging.debug(f"Debug, self.norm_specs is {self.norm_specs}")
            norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
            observation_detail = update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result, updated_param,
                                                           operation_region_dict)
            logging.debug(f"Step!!!DC Check fail. Observation detail: {observation_detail} "
                          f"with step number: {self.step_num}")
            observation = flatten_observation_w_region(observation_detail)
            logging.debug(f"Step!!!DC Check fail. Flatten Observation: {observation} with step number: {self.step_num}")
            observations = {agent: observation for agent in self.agents}
            rew = {a: -25 for a in self.agents}

            logging.info(f"Step!!!DC Check fail. Reward result: {rew} with step number: {self.step_num}")

            terminated = {a: False for a in self.agents}

            truncated = {a: False for a in self.agents}
            for agent_name in truncated:
                if self.step_num >= self.max_step:
                    truncated[agent_name] = True
                    self.truncateds.add(agent_name)
        # DC Check pass or not enabled
        else:
            # Run all simulations
            # Parse the updated param and generate the netlist

            # Create working directory
            working_dir_step = create_work_dir(self.run_root_dir)
            logging.info(f"Step!!! Working directory: {working_dir_step} with step number: {self.step_num}")

            update_netlist(working_dir_step, self.sim_config_dict, updated_param, self.unassigned_netlist_dir)

            # Run spectre simulation and normalize the result
            # Define a private function for retrying
            @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=self.zero_sim_result)
            def _run_simulation_with_retry():
                return run_dynamic_simulation(working_dir_step, self.sim_config_dict, self.zero_sim_result,
                                              self.sim_output_enable)

            sim_result, fail_tage = _run_simulation_with_retry()
            logging.info(f"Step!!!Simulation result: {sim_result} with step number: {self.step_num}")
            logging.debug(f"Debug, sim_result is {sim_result}")
            logging.debug(f"Debug, self.norm_specs is {self.norm_specs}")
            norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)

            # Generate observation
            # Add logic to avoid exception when no DC sim file
            try:
                working_dir_step_dc_passed = create_work_dir(self.run_root_dir)
                update_netlist(working_dir_step_dc_passed, self.dc_sim_config_dict, updated_param,
                               self.unassigned_netlist_dir)
                run_region_simulation(working_dir_step_dc_passed, self.dc_sim_config_dict, self.zero_sim_result,
                                      self.sim_output_enable)
                dc_raw_result_path = os.path.join(working_dir_step_dc_passed, "Region.raw/dcOpInfo.info")
                dc_result_path = os.path.join(working_dir_step_dc_passed, "Region.raw/dcOpInfo.info.encode")
                subprocess.run(f"psf {dc_raw_result_path} -o {dc_result_path}", shell=True)
                operation_region_dict = extract_operation_region_w_name(dc_result_path)
                # (working_dir_step_dc_passed)
            except Exception as e:
                logging.warning(f"Step Warning!!!: {e}. No DC sim file.")
                operation_region_dict = self.operation_region_dict_zero

            # Check operation_region_dict length vs self.operation_region_dict_zero length
            if len(operation_region_dict) != len(self.operation_region_dict_zero):
                logging.warning(f"Warning!!!: Operation region dict length {len(operation_region_dict)} does not match "
                                f"with operation_region_dict_zero length {len(self.norm_ideal_specs)}.")
                operation_region_dict = self.operation_region_dict_zero

            observation_detail = update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result, updated_param,
                                                           operation_region_dict)
            logging.debug(f"Step!!!Observation detail: {observation_detail} with step number: {self.step_num}")
            observation = flatten_observation_w_region(observation_detail)
            logging.debug(f"Step!!!Flatten Observation: {observation} with step number: {self.step_num}")

            # Share all observations
            observations = {agent: observation for agent in self.agents}

            # Calculate reward
            rew = {a: -25 for a in self.agents}
            # if fail_tage:
            #     rew_single = -25
            #     logging.warning(f"Step!!!Reward is given to min due to some simulation failed")
            # else:
            #     rew_single = cal_reward(self.ideal_specs, sim_result, self.norm_specs)
            rew_single = cal_reward(self.ideal_specs, sim_result, self.norm_specs)
            for agent_name in rew:
                rew[agent_name] = rew_single
            logging.info(f"Step!!!Reward result: {rew_single} with step number: {self.step_num}")

            # Determine termination or truncations
            terminated = {a: False for a in self.agents}
            for agent_name in terminated:
                if rew[agent_name] >= 0:
                    terminated[agent_name] = True
                    self.terminateds.add(agent_name)

            truncated = {a: False for a in self.agents}
            for agent_name in truncated:
                if self.step_num >= self.max_step:
                    truncated[agent_name] = True
                    self.truncateds.add(agent_name)

            # Delete working temp directory, if it exists
            # delete_work_dir(working_dir_step)

        info = {agent: {} for agent in self.agents}

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        logging.info(f"Step!!!terminated: {terminated} with step number: {self.step_num}")
        logging.info(f"Step!!!truncated: {truncated} with step number: {self.step_num}")

        step_data = {
            'step_num': self.step_num,
            'sim_result': sim_result,
            'updated_param': updated_param,
            'rew': rew
        }

        self.trajectory_data['steps_data'].append(step_data)

        with open(self.log_file_path, 'wb') as f:
            pickle.dump(self.trajectory_data, f)

        return observations, rew, terminated, truncated, info
