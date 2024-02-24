import os
import yaml
from collections import OrderedDict
import pickle
import datetime
import random
import shutil

from util.gen_action_sapce import gen_masked_action_space
# from util.gen_obs_space import gen_obs_space_extend, flatten_obs_space
# from util.gen_obs_space import gen_obs_space_w_type, flatten_obs_space_w_type
# from util.gen_obs_space import gen_obs_space_simple
from util.gen_obs_space import gen_obs_space_w_region, flatten_obs_space_w_region
from util.gen_param_space import gen_param_space
from util.util_func import create_work_dir
from util.assign_param2netlist import assign_param2netlist
from util.run_spectre_simulation import run_spectre_simulation
# from util.cal_reward import cal_reward_simple as cal_reward
from util.cal_reward import cal_reward
from util.generalize_config import generalize_config
from util.update_param import update_parameters
# from util.update_obs_space import update_obs_space, flatten_observation
# from util.update_obs_space import update_obs_space_simple
# from util.update_obs_space import update_obs_space_w_type, flatten_observation_w_type
from util.update_obs_space import update_obs_space_w_region, flatten_observation_w_region
from util.normlization import norm_ideal_spec, norm_sim_spec
from util.util_func import retry_decorator
from util.init_param import gen_init_param
from util.extract_device_param_value import extract_operation_region_w_name

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RllibAnalogDesignAutoEnv(MultiAgentEnv):
    def __init__(self, generalize=True,
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

        # Set normalization items
        self.norm_specs_file = config_folder_name + "/norm_specs.yaml"
        self.norm_specs_file = os.path.join(self.current_path, self.norm_specs_file)
        with open(self.norm_specs_file, 'r') as file:
            self.norm_specs = yaml.safe_load(file)

        # Get run file directory
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Set root directory based on home directory
        # Avoid run folder placing in /tmp directory when running on cluster
        # ~/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/ + self.run_root_dir
        self.run_root_dir = run_folder_name
        self.run_root_dir = os.path.join(file_dir, self.run_root_dir)
        if not os.path.exists(self.run_root_dir):
            raise ValueError(f"Root directory {self.run_root_dir} not found.")

        # Load config files
        self.agent_assign_config = config_folder_name + "/agent_assign.yaml"
        self.agent_assign_config = os.path.join(self.current_path, self.agent_assign_config)
        self.param_range_config = config_folder_name + "/param_range.yaml"
        self.param_range_config = os.path.join(self.current_path, self.param_range_config)
        self.sim_config = config_folder_name + "/simulation.yaml"
        self.sim_config = os.path.join(self.current_path, self.sim_config)
        self.predefined_init_param = config_folder_name + "/init_param.yaml"
        self.predefined_init_param = os.path.join(self.current_path, self.predefined_init_param)
        self.device_mask_config = config_folder_name + "/device_mask.yaml"
        self.device_mask_config = os.path.join(self.current_path, self.device_mask_config)

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

        # Create an empty dict for operation region

        self.operation_region_dict_zero = {}

        with open(self.param_range_config, 'r') as file:
            param_range = yaml.safe_load(file)
        for component, data in param_range.items():
            if component == 'other_variable':
                pass
            else:
                self.operation_region_dict_zero[component] = 0

        # RLlib config
        self.terminateds = set()
        self.truncateds = set()
        self._obs_space_in_preferred_format = True
        # self.observation_space = flatten_obs_space_w_type(gen_obs_space_w_type(
        #     self.sim_config, self.param_range_config, self.agent_assign_config))
        self.observation_space = flatten_obs_space_w_region(gen_obs_space_w_region(self.sim_config,
                                                                                   self.param_range_config,
                                                                                   self.agent_assign_config))
        # print(f"observation_space: {self.observation_space}")
        self._action_space_in_preferred_format = True
        self.action_space = gen_masked_action_space(action_mask, self.device_mask_config, self.agent_assign_config)
        # self.action_space = gen_action_space(self.agent_assign_config)
        # print(f"action_space: {self.action_space}")

        self.resetted = False

        super().__init__()

    def reset(self, *, seed=None, options=None):

        # Reset step number
        self.step_num = 0

        # Set the ideal specs based on the generalize flag
        self.ideal_specs = generalize_config(self.generalize, self.ideal_specs_path)
        print(f"Initialing!!!Generalize flag: {self.generalize}")
        print(f"Initialing!!!Ideal specs: {self.ideal_specs}")

        # Generate init param
        init_param = None
        working_dir = None
        exception_occurred = None
        if self.init_method == 'random':
            print(f"Initialing Checking!!!Init method: {self.init_method}")
            valid_init_param = False
            init_step = 0
            while not valid_init_param:
                try:
                    init_param = gen_init_param(self.init_method, self.predefined_init_param, self.action_mask,
                                                self.device_mask_config, self.param_space)
                    # Check the operation region of transistors with the given init params
                    working_dir = create_work_dir(self.run_root_dir)
                    print(f"Initialing Checking!!!Working directory: {working_dir} with init step number: {init_step}")

                    # Update DC Netlist File for checking init operation region
                    unassigned_netlist_file = "DC_parameterized.scs"
                    assigned_netlist_file = "DC.scs"
                    unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                    assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                    assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                    assign_param2netlist(init_param, unassigned_netlist_file_path, assigned_netlist_file_path,
                                         assigned_yaml_file_path)

                    with open(self.sim_config, 'r') as file:
                        sim_config = yaml.safe_load(file)

                    dc_sim_config = [item for item in sim_config if item['simulation_name'] == 'DC']
                    _ = run_spectre_simulation(working_dir, dc_sim_config, self.sim_output_enable)
                    dc_result_path = os.path.join(working_dir, "DC.raw/dcOpInfo.info.encode")
                    operation_region_dict = extract_operation_region_w_name(dc_result_path)
                    operation_region_list = list(operation_region_dict.values())
                    print(f"Initialing Checking!!!Operation region: {operation_region_list} with "
                          f"init step number: {init_step}")
                    # 0 cut-off, 1 triode, 2 saturation, 3 sub-th, 4 breakdown
                    # Check whether all transistors are in saturation/sub-threshold/triode region
                    valid_init_param = all(item in [1, 2, 3] for item in operation_region_list)
                    init_step += 1
                except Exception as e:
                    print(f"Warning!!!: {e}. Failed to generate init param with init step number: {init_step}")
                    exception_occurred = True
                    break
                # Delete working temp directory, if it exists
                finally:
                    try:
                        if os.path.exists(working_dir):
                            shutil.rmtree(working_dir)
                    except OSError as e:
                        print(f"Warning!!!: {e.strerror}. Directory {working_dir} does not exist or cannot be removed.")
                        pass
            if exception_occurred:
                init_param = gen_init_param('file', self.predefined_init_param, self.action_mask,
                                            self.device_mask_config, self.param_space)
        else:
            init_param = gen_init_param(self.init_method, self.predefined_init_param, self.action_mask,
                                        self.device_mask_config, self.param_space)

        # Generate working directory
        working_dir = create_work_dir(self.run_root_dir)
        print(f"Initialing!!!Working directory: {working_dir}")

        # Update Netlist File
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

        try:
            sim_result = run_spectre_simulation(working_dir, sim_config, self.sim_output_enable)
        # For avoid simulation error in init, use zero result instead.
        except Exception as e:
            print(f"Warning!!!: {e}. Simulation failed, use zero result instead.")
            sim_result = {}
            for sim in sim_config:
                sim_name = sim['simulation_name']
                sim_items = sim['simulation_item']
                sim_result[sim_name] = {item: 0.0 for item in sim_items}

        # For avoid simulation error in step, generate a default result with zero value but correct key in step method
        self.zero_sim_result = {k: {inner_k: 0.0 for inner_k in v} for k, v in sim_result.items()}

        # Normalize the current ideal specs
        print(f"Initialing!!!Ideal specs: {self.ideal_specs}")
        self.norm_ideal_specs = norm_ideal_spec(self.ideal_specs, self.norm_specs)
        # print(f"Initialing!!!Normalized ideal specs: {self.norm_ideal_specs}")

        # Normalize the current simulation specs
        print(f"Initialing!!!Simulation result: {sim_result}")
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
        # print(f"Initialing!!!Normalized simulation result: {norm_sim_result}")

        # Generate region observation
        try:
            dc_result_path = os.path.join(working_dir, "DC.raw/dcOpInfo.info.encode")
            operation_region_dict = extract_operation_region_w_name(dc_result_path)
        except Exception as e:
            print(f"Warning!!!: {e}. No DC sim file.")
            operation_region_dict = self.operation_region_dict_zero

        # Check operation_region_dict length vs self.operation_region_dict_zero length
        if len(operation_region_dict) != len(self.operation_region_dict_zero):
            print(f"Warning!!!: Operation region dict length {len(operation_region_dict)} does not match with "
                  f"operation_region_dict_zero length {len(self.norm_ideal_specs)}.")
            operation_region_dict = self.operation_region_dict_zero

        # Generate observation
        # observation = update_obs_space_simple(self.ideal_specs, norm_sim_result, init_param)
        # observation_detail = update_obs_space_w_type(self.norm_ideal_specs, norm_sim_result, init_param,
        #                                              self.param_range_config)
        observation_detail = update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result, init_param,
                                                       operation_region_dict)
        # print(f"Initialing!!!Observation result: {observation_detail}")
        # observation = flatten_observation_w_type(observation_detail)
        observation = flatten_observation_w_region(observation_detail)

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
        try:
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
        except OSError as e:
            print(f"Warning!!!: {e.strerror}. Directory {working_dir} does not exist or cannot be removed.")
            pass

        return observations, info

    def step(self, action_dict):

        # Update step number
        self.step_num += 1

        operation_region_dict = {}

        # print(f"Updated action: {action_dict} with step number: {self.step_num}")

        # Create working directory
        working_dir = create_work_dir(self.run_root_dir)
        print(f"Step!!! Working directory: {working_dir} with step number: {self.step_num}")

        with open(self.device_mask_config, 'r') as file:
            device_mask = yaml.safe_load(file)

        if self.action_mask:
            for key in device_mask:
                for agent_key in action_dict:
                    if key in action_dict[agent_key]:
                        new_values = device_mask[key]
                        for new_key in new_values:
                            new_value = action_dict[agent_key][key]
                            action_dict[agent_key][new_key] = new_value

        # print(f"Step!!!Actions: {action_dict} with step number: {self.step_num}")
        # Flatten all actions
        all_action_flatten = OrderedDict()
        for group in action_dict.values():
            for key, value in group.items():
                all_action_flatten[key] = value
        # print(f"Step!!!Flatten actions: {all_action_flatten}")

        # Update param with new action
        # print(f"Step!!!Current param index: {self.cur_param} with step number: {self.step_num}")
        # print(f"Step!!!Previous param: {self.cur_param} with step number: {self.step_num}")
        updated_param = update_parameters(all_action_flatten, self.cur_param, self.param_range_config)
        # print(f"Step!!!Updated param: {updated_param} with step number: {self.step_num}")

        # Update current param
        self.cur_param = updated_param

        # Run DC check firstly and only once. If dc_check is True. If DC check failed, return zero sim result and -10
        # reward. End the episode.
        valid_param = None
        if self.dc_check:
            try:
                # Update DC Netlist File for checking operation region
                unassigned_netlist_file = "DC_parameterized.scs"
                assigned_netlist_file = "DC.scs"
                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(updated_param, unassigned_netlist_file_path, assigned_netlist_file_path,
                                     assigned_yaml_file_path)

                with open(self.sim_config, 'r') as file:
                    sim_config = yaml.safe_load(file)

                dc_sim_config = [item for item in sim_config if item['simulation_name'] == 'DC']
                _ = run_spectre_simulation(working_dir, dc_sim_config, self.sim_output_enable)
                dc_result_path = os.path.join(working_dir, "DC.raw/dcOpInfo.info.encode")
                operation_region_dict = extract_operation_region_w_name(dc_result_path)
                operation_region_list = list(operation_region_dict.values())
                print(f"Step!!!Operation region: {operation_region_list} with step number: {self.step_num}")
                # 0 cut-off, 1 triode, 2 saturation, 3 sub-th, 4 breakdown
                # Check whether all transistors are in saturation/sub-threshold/triode region
                valid_param = all(item in [1, 2, 3] for item in operation_region_list)
                # Check operation_region_dict length vs self.operation_region_dict_zero length
                if len(operation_region_dict) != len(self.operation_region_dict_zero):
                    print(f"Warning!!!: Operation region dict length {len(operation_region_dict)} does not match with "
                          f"operation_region_dict_zero length {len(self.norm_ideal_specs)}.")
                    operation_region_dict = self.operation_region_dict_zero
                    valid_param = False
            except Exception as e:
                print(f"Warning!!!: {e}. Failed to run DC check with step number: {self.step_num}")
                operation_region_dict = self.operation_region_dict_zero
                valid_param = False

        if self.dc_check and not valid_param:
            print(f"Step!!!Param failed DC check with step number: {self.step_num}")
            sim_result = self.zero_sim_result
            norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
            observation_detail = update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result, updated_param,
                                                           operation_region_dict)
            observation = flatten_observation_w_region(observation_detail)
            observations = {agent: observation for agent in self.agents}
            rew = {a: -10 for a in self.agents}

            print(f"Step!!!DC Check fail.Reward result: {rew} with step number: {self.step_num}")

            terminated = {a: False for a in self.agents}

            truncated = {a: False for a in self.agents}
            for agent_name in truncated:
                if self.step_num >= self.max_step:
                    truncated[agent_name] = True
                    self.truncateds.add(agent_name)
        else:
            # Run all simulations
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
            # Define a private function for retrying
            @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=self.zero_sim_result)
            def _run_simulation_with_retry():
                return run_spectre_simulation(working_dir, sim_config, self.sim_output_enable)

            sim_result = _run_simulation_with_retry()
            print(f"Step!!!Simulation result: {sim_result} with step number: {self.step_num}")
            norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
            # print(f"Step!!!Normalized simulation result: {norm_sim_result} with step number: {self.step_num}")

            # Generate observation
            # observation = update_obs_space_simple(self.ideal_specs, norm_sim_result, updated_param)
            # observation_detail = update_obs_space_w_type(self.norm_ideal_specs, norm_sim_result, updated_param,
            #                                              self.param_range_config)

            # Add logic to avoid exception when no DC sim file
            try:
                dc_result_path = os.path.join(working_dir, "DC.raw/dcOpInfo.info.encode")
                operation_region_dict = extract_operation_region_w_name(dc_result_path)
            except Exception as e:
                print(f"Step Warning!!!: {e}. No DC sim file.")
                operation_region_dict = self.operation_region_dict_zero

            # Check operation_region_dict length vs self.operation_region_dict_zero length
            if len(operation_region_dict) != len(self.operation_region_dict_zero):
                print(f"Warning!!!: Operation region dict length {len(operation_region_dict)} does not match with "
                      f"operation_region_dict_zero length {len(self.norm_ideal_specs)}.")
                operation_region_dict = self.operation_region_dict_zero

            observation_detail = update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result, updated_param,
                                                           operation_region_dict)
            # print(f"Step!!!Observation result: {observation_detail} with step number: {self.step_num}")
            # observation = flatten_observation_w_type(observation_detail)
            observation = flatten_observation_w_region(observation_detail)

            # Share all observations
            observations = {agent: observation for agent in self.agents}
            # print(f"Step!!!Observations result: {observations} with step number: {self.step_num}")

            # Store the observation
            # with open(os.path.join(working_dir, "result.yaml"), 'w') as file:
            #     yaml.dump(observation, file)

            # print("Step!!!self.ideal_specs: ", self.ideal_specs)
            # print("Step!!!observation_detail: ", observation_detail)

            # Calculate reward
            rew = {a: -10 for a in self.agents}
            rew_single = cal_reward(self.ideal_specs, sim_result)
            for agent_name in rew:
                rew[agent_name] = rew_single
            print(f"Step!!!Reward result: {rew_single} with step number: {self.step_num}")

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

        info = {agent: {} for agent in self.agents}

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        print(f"Step!!!terminated: {terminated} with step number: {self.step_num}")
        print(f"Step!!!truncated: {truncated} with step number: {self.step_num}")

        step_data = {
            'step_num': self.step_num,
            'sim_result': sim_result,
            'updated_param': updated_param,
            'rew': rew
        }

        self.trajectory_data['steps_data'].append(step_data)

        with open(self.log_file_path, 'wb') as f:
            pickle.dump(self.trajectory_data, f)

        # Delete working temp directory, if it exists
        try:
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
        except OSError as e:
            print(f"Warning!!!: {e.strerror}. Directory {working_dir} does not exist or cannot be removed.")
            pass

        return observations, rew, terminated, truncated, info
