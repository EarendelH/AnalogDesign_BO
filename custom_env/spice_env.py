import copy
import os
import yaml
from collections import OrderedDict
import logging
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Any, Union
import importlib
import pickle

from util.gen_action_space import gen_masked_continuous_action_space
from util.gen_obs_space import gen_obs_space_w_region, flatten_obs_space_w_region
from util.gen_obs_space import gen_obs_space, flatten_obs_space
from util.action2param import action2param
from util.gen_param_space import gen_param_space
from util.util_func import create_work_dir, create_corner_work_dir
from util.assign_param2netlist import update_netlist
from util.run_spectre_simulation import run_dynamic_simulation, run_region_simulation
from util.generalize_config import generalize_config
from util.update_obs_space import update_obs_space_w_region, flatten_observation_w_region
from util.update_obs_space import update_obs_space, flatten_observation
from util.normlization import norm_ideal_spec, norm_sim_spec
from util.util_func import retry_decorator
from util.gen_init_param import gen_init_param
from util.device_mask import masked_action_dict_mapping


class RllibAnalogDesignAutoEnv(MultiAgentEnv):

    def _validate_config(self, config):
        """验证输入配置"""
        expected_params = {
            'generalize': bool,
            'max_step': int,
            'netlist_folder_name': str,
            'specs_folder_name': str,
            'config_folder_name': str,
            'run_folder_name': str,
            'sim_output': bool,
            'init_method': str,
            'corner_sim': bool,
            'dc_check': bool,
            'region_extract': bool,
            'dynamic_queue': bool,
            'log_level': str,
            'reward_func': str
        }
        for param, expected_type in expected_params.items():
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(config[param], expected_type):
                raise ValueError(f"Invalid type for {param}. Expected {expected_type}, got {type(config[param])}")

    def _set_basic_attributes(self, config):
        """批量设置基础属性"""
        for key in config:
            setattr(self, key, config[key])

        # 设置标志属性
        flag_mappings = {
            'continue_steps_enable': 'continue_steps_enable'
        }
        for dest, src in flag_mappings.items():
            setattr(self, dest, getattr(self, src, False))

    def _setup_paths(self):
        """配置所有路径"""
        self.current_path = os.getcwd()
        self.home_dir = os.path.expanduser("~")

        # 主运行目录
        self.run_root_dir = os.path.join(
            self.home_dir,
            "AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env",
            self.run_folder_name
        )
        os.makedirs(self.run_root_dir, exist_ok=True)

        # 配置文件路径
        self.config_files = {
            'agent_assign_config': "agent_assign.yaml",
            'param_range_config': "param_range.yaml",
            'sim_config': "simulation.yaml",
            'dc_sim_config': "simulation_region.yaml",
            'predefined_init_param': "init_param.yaml",
            'device_mask_config': "device_mask.yaml",
            'norm_specs_file': "norm_specs.yaml",
            'generalize_specs_config': "generalize_specs.yaml",
        }

        config_folder = os.path.join(self.current_path, 'config', self.config_folder_name)
        for attr, filename in self.config_files.items():
            setattr(self, attr, os.path.join(config_folder, filename))

        self.ideal_specs_path = os.path.join(self.current_path, 'ideal_specs', self.specs_folder_name)
        self.unassigned_netlist_dir = self.netlist_folder_name
        self.unassigned_netlist_dir = os.path.join(self.current_path, 'netlist_template', self.unassigned_netlist_dir)

    def _load_all_configs(self):
        """加载所有YAML配置"""
        self.sim_config_dict = self._load_yaml_config(self.sim_config)
        self.norm_specs = self._load_yaml_config(self.norm_specs_file)
        self.param_range_dict = self._load_yaml_config(self.param_range_config)
        self.agent_assign_dict = self._load_yaml_config(self.agent_assign_config)
        self.device_mask_dict = self._load_yaml_config(self.device_mask_config)
        self.dc_sim_config_dict = self._load_yaml_config(self.dc_sim_config)
        self.generalize_specs_config_dict = self._load_yaml_config(self.generalize_specs_config)

    def _load_yaml_config(self, file_path: str) -> dict:
        """通用YAML加载方法"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config {file_path}: {e}")
            raise

    def _setup_logging(self):
        """配置日志系统"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _initialize_spaces(self):
        """初始化观察和动作空间"""
        # 生成参数空间
        self.param_space = copy.deepcopy(gen_param_space(self.param_range_dict))

        # 生成观察空间
        obs_generator = gen_obs_space_w_region if self.region_extract else gen_obs_space
        flatten_obs = flatten_obs_space_w_region if self.region_extract else flatten_obs_space
        self.observation_space = flatten_obs(obs_generator(
            self.sim_config_dict,
            self.param_range_dict,
            self.agent_assign_dict
        ))

        # 生成动作空间
        self.action_space = gen_masked_continuous_action_space(
            self.device_mask_dict,
            self.agent_assign_dict
        )

    def _initialize_state_variables(self):
        """初始化状态变量"""

        self.resetted = None
        self.trajectory_data = None
        self.log_file_path = None
        self.log_file_name = None
        self.zero_sim_result = None
        self.step_num = None
        self.norm_ideal_specs = None
        self.ideal_specs = None
        self.cur_param = None
        self.max_step = int(self.max_step)
        self.action_mask = True

        self.continue_steps = 4  # Number of steps to continue after positive reward
        self.steps_after_positive_reward = 0
        self.had_positive_reward = False

        self.operation_region_dict_zero = {
            comp: 0 for comp in self.param_range_dict
            if comp != 'other_variable'
        }
        self.possible_agents = list(self.agent_assign_dict.keys())
        self.agents = self.possible_agents
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

    def _import_reward_function(self):
        try:
            reward_module = importlib.import_module('util.cal_reward')
            self.cal_reward = getattr(reward_module, self.reward_func)
        except (ImportError, AttributeError) as e:
            logging.error(f"Error importing reward function '{self.reward_func}': {e}")
            raise

    def __init__(self, config: Dict[str, Any]):
        # 配置验证和基础设置
        self._validate_config(config)
        self._set_basic_attributes(config)

        # 路径配置
        self._setup_paths()

        # 日志配置
        self._setup_logging()

        # 加载所有YAML配置
        self._load_all_configs()

        # 初始化环境空间和参数
        self._initialize_spaces()
        self._initialize_state_variables()

        # Import Reward Func
        self._import_reward_function()

        super().__init__()

    def _initialize_reset_variables(self):
        self.steps_after_positive_reward = 0
        self.had_positive_reward = False

        self.step_num = 0

        # For avoid simulation error in step, generate a default result with zero value but correct key in step method
        self.zero_sim_result = {}
        for sim in self.generalize_specs_config_dict:
            specs_tmp_dict = {}
            for specs_item in self.generalize_specs_config_dict[sim]:
                if self.generalize_specs_config_dict[sim][specs_item]['objective'] == 'max':
                    specs_tmp_dict[specs_item] = 0.0
                if self.generalize_specs_config_dict[sim][specs_item]['objective'] == 'min':
                    specs_tmp_dict[specs_item] = 100.0
                if self.generalize_specs_config_dict[sim][specs_item]['objective'] == 'range':
                    specs_tmp_dict[specs_item] = 100.0
            self.zero_sim_result[sim] = specs_tmp_dict
        logging.info(f"Initialing!!!Zero sim result: {self.zero_sim_result}")

        # Set the ideal specs based on the generalize flag
        self.ideal_specs = generalize_config(self.generalize, self.ideal_specs_path)

    def _run_region_simulation_check(self, dir_suffix: str, param: dict) -> dict:
        try:
            working_dir = create_work_dir(self.run_root_dir, dir_suffix)
            update_netlist(working_dir, self.dc_sim_config_dict, param, self.unassigned_netlist_dir)
            operation_region_dict = copy.deepcopy(
                run_region_simulation(working_dir, self.dc_sim_config_dict, self.sim_output))

            # 检查结果长度一致性
            if len(operation_region_dict) != len(self.operation_region_dict_zero):
                logging.warning(
                    f"Operation region dict length mismatch. Expected {len(self.operation_region_dict_zero)}, Got {len(operation_region_dict)}")
                return copy.deepcopy(self.operation_region_dict_zero)

            return operation_region_dict
        except Exception as e:
            logging.warning(f"Region simulation failed: {str(e)}")
            return copy.deepcopy(self.operation_region_dict_zero)

    def _process_simulation_result(self, sim_result: dict, param: dict,
                                   operation_region_dict: dict = None) -> tuple:
        """统一处理模拟结果的后处理流程"""
        # 结果规范化
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
        logging.debug(f"Normalized simulation result: {norm_sim_result}")

        # 生成观测空间
        if self.region_extract:
            observation_detail = copy.deepcopy(
                update_obs_space_w_region(
                    self.norm_ideal_specs,
                    norm_sim_result,
                    param,
                    operation_region_dict  # 仅在region_extract时传递第四个参数
                )
            )
            observation = copy.deepcopy(flatten_observation_w_region(observation_detail))
        else:
            observation_detail = copy.deepcopy(
                update_obs_space(  # 非region_extract时使用三参数版本
                    self.norm_ideal_specs,
                    norm_sim_result,
                    param
                )
            )
            observation = copy.deepcopy(flatten_observation(observation_detail))

        # 构建多智能体观测
        observations = {agent: observation for agent in self.agents}

        # 计算奖励
        rew = self.cal_reward(self.ideal_specs, sim_result, self.norm_specs)

        return observations, sim_result, rew

    def _run_simulation(self, working_dir: str, param: dict,
                        operation_region_dict: dict = None) -> tuple:
        """统一执行模拟流程"""

        # 带重试机制的模拟运行
        @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=self.zero_sim_result)
        def _run_with_retry():
            return run_dynamic_simulation(
                working_dir,
                self.sim_config_dict,
                self.zero_sim_result,
                self.sim_output,
                self.dynamic_queue
            )

        try:
            sim_result = copy.deepcopy(_run_with_retry())
        except Exception as e:
            logging.warning(f"Simulation failed: {e}, using zero result")
            sim_result = copy.deepcopy(self.zero_sim_result)

        return self._process_simulation_result(sim_result, param, operation_region_dict)

    def _handle_corner_simulations(self, main_dir: str, param: dict,
                                   operation_region_dict: dict) -> dict:
        """统一处理多角落模拟"""
        corner_results = {}
        for corner in ['ff', 'fs', 'sf', 'ss']:
            corner_dir = create_corner_work_dir(main_dir, corner)
            update_netlist(corner_dir, self.sim_config_dict, param, self.unassigned_netlist_dir, corner)
            obs, sim_result, rew = self._run_simulation(
                corner_dir,
                param,
                operation_region_dict
            )
            corner_results[corner] = {
                'working_dir': corner_dir,
                'sim_result': sim_result,
                'reward': rew
            }
        return corner_results

    def reset(self, *, seed=None, options=None):

        self._initialize_reset_variables()

        reset_operation_region_dict = None

        logging.info(f"Initialing!!!Generalize flag: {self.generalize}")
        logging.info(f"Initialing!!!Ideal specs: {self.ideal_specs}")

        init_param = gen_init_param(self.init_method, self.predefined_init_param, True,
                                    self.device_mask_dict, self.param_space)
        logging.debug(f"Initialing!!!Init param: {init_param}")

        # Add _init in the path
        working_dir_reset = create_work_dir(self.run_root_dir, 'init')

        # Update Netlist File
        update_netlist(working_dir_reset, self.sim_config_dict, init_param, self.unassigned_netlist_dir)

        # Normalize the current ideal specs
        self.norm_ideal_specs = norm_ideal_spec(self.ideal_specs, self.norm_specs)
        logging.info(f"Initialing!!!Ideal specs: {self.ideal_specs}")
        logging.debug(f"Initialing!!!Normalized ideal specs: {self.norm_ideal_specs}")

        if self.region_extract:
            reset_operation_region_dict = self._run_region_simulation_check('init_dc', init_param)
            logging.debug(f"Initial!!! operation regions: {reset_operation_region_dict}")

        observations, sim_result, rew = self._run_simulation(
            working_dir_reset,
            init_param,
            reset_operation_region_dict if self.region_extract else None
        )

        # Normalize the current simulation specs
        logging.info(f"Initialing!!!Simulation result: {sim_result}")
        logging.debug(f"Initialing!!!Ideal specs: {self.ideal_specs}")
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)
        logging.debug(f"Initialing!!!Normalized simulation result: {norm_sim_result}")

        # Generate observation
        observations, sim_result, rew = self._run_simulation(
            working_dir_reset,
            init_param,
            reset_operation_region_dict if self.region_extract else None
        )

        # Test Rew func
        rew = self.cal_reward(self.ideal_specs, sim_result, self.norm_specs)
        logging.info(f"Debug!!!Resetting!!!Reward result: {rew}")

        self.cur_param = copy.deepcopy(init_param)

        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()

        info = {agent: {} for agent in self.agents}

        # Save init step to pickle file
        step_data = {
            'param': init_param,
            'sim_result': sim_result,
            'reward': rew,
            'corner': 'tt'
        }
        pickle_path = os.path.join(working_dir_reset, 'result.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(step_data, f)

        return observations, info

    def step(self, action_dict):
        logging.debug(f"Step!!!Action dict: {action_dict}")
        # Update step number
        self.step_num += 1
        rew_single = -10

        operation_region_dict = None
        valid_param = None

        step_action_dict = copy.deepcopy(action_dict)
        if self.device_mask_dict:
            all_action_flatten = copy.deepcopy(masked_action_dict_mapping(self.device_mask_dict, step_action_dict))
            logging.debug(f"Step!!!Action dict w/ device mask: {all_action_flatten}")
        else:
            mapped_step_action_dict = copy.deepcopy(step_action_dict)
            logging.debug(f"Step!!!Action dict w/o device mask: {mapped_step_action_dict}")
            # Flatten all actions
            all_action_flatten = OrderedDict()
            for group in mapped_step_action_dict.values():
                for key, value in group.items():
                    all_action_flatten[key] = value

        # Update param with new action
        logging.debug(f"Step!!!Action Mask: {self.action_mask}")
        logging.debug(f"Step!!!Device Mask: {self.device_mask_dict}")
        logging.debug(f"Step!!!All action flatten: {all_action_flatten}")
        logging.debug(f"Step!!!Current param: {self.cur_param}")
        logging.debug(f"Step!!!Param range: {self.param_range_dict}")
        updated_param = copy.deepcopy(action2param(self.action_mask, self.device_mask_dict, all_action_flatten,
                                                   self.param_range_dict))
        logging.info(f"Step!!!Updated param: {updated_param} with step number: {self.step_num}")

        # Update current param
        self.cur_param = copy.deepcopy(updated_param)

        # Run DC check firstly and only once. If dc_check is True. If DC check failed, return zero sim result and -10
        # reward. End the episode.

        if self.region_extract:
            operation_region_dict = self._run_region_simulation_check('dc', updated_param)
            operation_region_list = list(operation_region_dict.values())
            # 0 cut-off, 1 triode, 2 saturation, 3 sub-th, 4 breakdown
            # Check whether all transistors are in saturation/sub-threshold/triode region
            valid_param = all(item in [1, 2, 3] for item in operation_region_list)
            logging.info(f"Step operation regions: {operation_region_list}")

        # DC check fail condition
        if self.region_extract and self.dc_check and not valid_param:
            logging.info(f"Step!!! Region_extract & DC_Check is enable and the param is not passed with dc_check"
                         f" with step number: {self.step_num}")
            sim_result = copy.deepcopy(self.zero_sim_result)
            logging.debug(f"Debug, sim_result is {sim_result}")
            logging.debug(f"Debug, self.norm_specs is {self.norm_specs}")
            norm_sim_result = copy.deepcopy(norm_sim_spec(sim_result, self.norm_specs))
            observation_detail = copy.deepcopy(update_obs_space_w_region(self.norm_ideal_specs, norm_sim_result,
                                                                         updated_param, operation_region_dict))
            logging.debug(f"Step!!!DC Check fail. Observation detail: {observation_detail} "
                          f"with step number: {self.step_num}")
            observation = copy.deepcopy(flatten_observation_w_region(observation_detail))
            logging.debug(
                f"Step!!!DC Check fail. Flatten Observation: {observation} with step number: {self.step_num}")
            observations = {agent: observation for agent in self.agents}
            single_rew = -10

            logging.info(f"Step!!!Region_extract & DC_Check is enable and the param is not passed with dc_check."
                         f" Reward result: {single_rew} with step number: {self.step_num}")

        # DC Check pass or not enabled. Run all simulations
        else:
            working_dir_step_tt = create_work_dir(self.run_root_dir, 'tt')
            update_netlist(working_dir_step_tt, self.sim_config_dict, updated_param, self.unassigned_netlist_dir)
            observations_tt, sim_result_tt, rew_single_tt = self._run_simulation(
                working_dir_step_tt,
                updated_param,
                operation_region_dict if self.region_extract else None
            )

            if self.corner_sim and rew_single_tt >= 0:
                corner_results = self._handle_corner_simulations(working_dir_step_tt, updated_param, operation_region_dict)
                corner_results['tt'] = {
                    'working_dir': working_dir_step_tt,
                    'sim_result': sim_result_tt,
                    'reward': rew_single_tt
                }

                # Extract the min reward from all corners and replace the reward
                rew_single_min = min([corner_results[corner]['rew_single'] for corner in corner_results])
                if rew_single_min < 0:
                    rew_single_min = 10
                for corner in corner_results:
                    corner_results[corner]['rew_single'] = rew_single_min
                    step_data = {
                        'param': updated_param,
                        'sim_result': corner_results[corner]['sim_result'],
                        'reward': corner_results[corner]['rew_single'],
                        'corner': corner
                    }
                    pickle_path = os.path.join(corner_results[corner]['working_dir_step'], 'result.pkl')
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(step_data, f)

                observations = observations_tt
                rew_single = rew_single_min

            else:
                working_dir_step = working_dir_step_tt
                observations = observations_tt
                sim_result = sim_result_tt
                rew_single = rew_single_tt

                step_data = {
                    'param': updated_param,
                    'sim_result': sim_result,
                    'reward': rew_single,
                    'corner': 'tt'
                }
                pickle_path = os.path.join(working_dir_step, 'result.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(step_data, f)

        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}

        rew = {a: -10 for a in self.agents}
        for agent_name in rew:
            rew[agent_name] = rew_single

        if self.continue_steps_enable:
            if rew_single > 0 and not self.had_positive_reward:
                self.had_positive_reward = True
                self.steps_after_positive_reward = 0
            elif self.had_positive_reward:
                self.steps_after_positive_reward += 1

            episode_over = False
            if self.had_positive_reward and (
                    self.steps_after_positive_reward >= self.continue_steps or self.step_num >= self.max_step):
                episode_over = True
            elif self.step_num >= self.max_step:
                episode_over = True

            if episode_over:
                for agent_name in terminated:
                    if self.had_positive_reward:
                        terminated[agent_name] = True
                        self.terminateds.add(agent_name)
                    else:
                        truncated[agent_name] = True
                        self.truncateds.add(agent_name)

        else:
            if rew_single > 0:
                for agent_name in terminated:
                    terminated[agent_name] = True
                    self.terminateds.add(agent_name)
            if self.step_num >= self.max_step:
                for agent_name in truncated:
                    truncated[agent_name] = True
                    self.truncateds.add(agent_name)

            # Delete working temp directory, if it exists
            # delete_work_dir(working_dir_step)

        info = {agent: {} for agent in self.agents}

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        logging.info(f"Step!!!terminated: {terminated} with step number: {self.step_num}")
        logging.info(f"Step!!!truncated: {truncated} with step number: {self.step_num}")

        # step_data = {
        #     'step_num': self.step_num,
        #     'sim_result': sim_result,
        #     'updated_param': updated_param,
        #     'rew': rew
        # }
        #
        # self.trajectory_data['steps_data'].append(step_data)
        #
        # with open(self.log_file_path, 'wb') as f:
        #     pickle.dump(self.trajectory_data, f)

        return observations, rew, terminated, truncated, info

    def validate_input(self, config: Dict[str, Any]) -> None:
            for param, expected_type in self.expected_params.items():
                if param not in config:
                    raise ValueError(f"Missing required parameter: {param}")

                value = config[param]
                if not isinstance(value, expected_type):
                    raise ValueError(f"Invalid type for {param}. Expected {expected_type}, got {type(value)}")

                if param == 'init_method' and value not in ['file', 'half', 'random', 'mixed']:
                    raise ValueError(
                        f"Invalid value for init_method. Expected one of ['file', 'half', 'random', 'mixed'], got {value}")

                if param == 'log_level' and value not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    raise ValueError(
                        f"Invalid value for log_level. Expected one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], "
                        f"got {value}")

            for key in config:
                if key not in self.expected_params:
                    raise ValueError(f"Unexpected parameter: {key}")

def step_simulation_process(working_dir_step, region_extract_tag, valid_param, step_num, dc_check_tag,
                            sim_config_dict,
                            updated_param, unassigned_netlist_dir, zero_sim_result, sim_output_tag,
                            dynamic_queue_tag, norm_specs, norm_ideal_specs, agents, ideal_specs, cal_reward,
                            corner_tag, operation_region_dict=None):
    sim_result = None

    # Generate working directory
    os.makedirs(working_dir_step, exist_ok=True)

    if region_extract_tag:
        if operation_region_dict is None:
            logging.error("Error: region_extract is True but operation_region_dict is None.")
            raise ValueError("operation_region_dict is None when region_extract is True")

    # Create working directory
    if valid_param:
        logging.info(f"Step!!!Working directory: {working_dir_step} with step number: {step_num} "
                     f"with region_extract & dc_check enabled and valid param.")
    else:
        logging.info(f"Step!!!Working directory: {working_dir_step} with step number: {step_num} "
                     f"with region_extract: {region_extract_tag} and dc_check: {dc_check_tag}")

    update_netlist(working_dir_step, sim_config_dict, updated_param, unassigned_netlist_dir, corner_tag)

    # Run spectre simulation and normalize the result
    # Define a private function for retrying
    @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=zero_sim_result)
    def _run_simulation_with_retry():
        return run_dynamic_simulation(working_dir_step, sim_config_dict, zero_sim_result,
                                      sim_output_tag, dynamic_queue_tag)

    try:
        sim_result = copy.deepcopy(_run_simulation_with_retry())
    except Exception as e:
        logging.warning(f"Step Warning!!!: {e}. sim_result is {sim_result}."
                        f" Simulation failed, use zero result instead.")
        sim_result = copy.deepcopy(zero_sim_result)

    logging.info(f"Step!!!Simulation result: {sim_result} with step number: {step_num} in corner: {corner_tag}")
    logging.debug(f"Debug, sim_result is {sim_result}")
    logging.debug(f"Debug, self.norm_specs is {norm_specs}")
    norm_sim_result = norm_sim_spec(sim_result, norm_specs)

    if region_extract_tag:
        observation_detail = update_obs_space_w_region(norm_ideal_specs, norm_sim_result, updated_param,
                                                       operation_region_dict)
        observation = copy.deepcopy(flatten_observation_w_region(observation_detail))
    else:
        observation_detail = update_obs_space(norm_ideal_specs, norm_sim_result, updated_param)
        observation = copy.deepcopy(flatten_observation(observation_detail))
    logging.debug(f"Step!!!Observation detail: {observation_detail} with step number: {step_num}")
    logging.debug(f"Step!!!Flatten Observation: {observation} with step number: {step_num}")

    # Share all observations
    observations = {agent: observation for agent in agents}

    # Calculate reward
    rew_single = cal_reward(ideal_specs, sim_result, norm_specs)
    logging.info(f"Step!!!Reward result: {rew_single} with step number: {step_num}")

    return observations, sim_result, rew_single