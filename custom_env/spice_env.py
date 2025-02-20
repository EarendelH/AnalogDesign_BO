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

    def __init__(self, config: Dict[str, Any]):
        """Initialize the analog design automation environment.

        Args:
            config: Configuration dictionary containing environment parameters
        """
        # Configuration validation and basic setup

        self._validate_config(config)
        self._set_basic_attributes(config)

        # Path configuration
        self._setup_paths()

        # Logging configuration
        self._setup_logging()

        # Load all YAML configurations
        self._load_all_configs()

        # Initialize environment spaces and parameters
        self._initialize_spaces()
        self._initialize_state_variables()

        # Import reward function dynamically
        self._import_reward_function()

        super().__init__()

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.

        Returns:
            tuple: (observations, info) containing initial observations
                   and empty info dict
        """
        # Initialize reset-related variables
        self._initialize_reset_variables()

        self.logger.info("Initializing reset process...")
        self.logger.debug(f"Generalize flag: {self.generalize}")
        self.logger.debug(f"Ideal specs config: {self.ideal_specs_path}")

        # Load specifications
        self._load_ideal_specs()

        # Generate initial parameters
        init_param = self._generate_initial_parameters()
        self.cur_param = copy.deepcopy(init_param)

        self.logger.info(f"Generated initial parameters: {init_param}")
        self.logger.debug(f"Parameter space: {self.param_space}")

        # Setup simulation environment
        working_dir = self._setup_reset_environment(init_param)

        # Run initial DC simulation for region check
        operation_region = self._run_initial_dc_simulation(init_param)

        # Execute initial simulation
        observations, sim_result, rew = self._run_simulation(
            working_dir,
            init_param,
            operation_region if self.region_extract else None
        )
        self.logger.info(f"Initial simulation result: {sim_result}")
        self.logger.info(f"Initial reward: {rew}")

        # Save reset data
        self._save_reset_data(working_dir, init_param, sim_result, rew)

        # Reset environment state
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()

        return observations, {agent: {} for agent in self.agents}

    def step(self, action_dict):
        """Execute one environment step using provided actions.

        Args:
            action_dict: Dictionary of actions from all agents

        Returns:
            tuple: (observations, rewards, terminateds, truncateds, infos)
        """
        # Increment step counter
        self.step_num += 1

        self.logger.info(f"Starting step {self.step_num}")
        self.logger.debug(f"Raw action dict: {action_dict}")

        # 1. Convert actions to parameters
        updated_param, all_actions = self._convert_actions_to_params(copy.deepcopy(action_dict))
        self.logger.debug(f"Converted actions: {all_actions}")
        self.logger.info(f"Updated parameter: {updated_param}")
        self.cur_param = copy.deepcopy(updated_param)

        # 2. Perform DC check
        region_info, dc_valid = self._perform_step_dc_check(updated_param)

        # 3. Handle DC check failure
        if self.region_extract and self.dc_check and not dc_valid:
            observations, rew_single = self._handle_dc_check_failure(updated_param, region_info)
        else:
            # 4. Run main simulation
            observations, sim_result, rew_single, main_dir = self._run_main_simulation(updated_param, region_info)
            self.logger.info(f"Running main simulation in {main_dir}")
            self.logger.debug(f"Simulation config: {self.sim_config_dict}")

            # 5. Process corner simulations
            if self.corner_sim and rew_single >= 0:
                self.logger.info(f"Positive reward {rew_single} in step {self.step_num}, running corner simulations...")
                tt_result = {
                    'sim_result': sim_result,
                    'reward': rew_single
                }
                rew_single = self._process_corner_results(
                    main_dir,  # Pass main directory
                    updated_param,
                    region_info,
                    tt_result  # Pass TT simulation result
                )
            # 6. Save main simulation data (non-corner case)
            else:
                self._save_step_data(main_dir, updated_param, sim_result, rew_single)

        # 7. Calculate final rewards
        rewards = self._assign_rewards(rew_single)

        # 8. Determine termination conditions
        terminated, truncated = self._determine_termination(rew_single)

        # 9. Log final state
        logging.info(f"Step {self.step_num} terminated: {terminated}")
        logging.info(f"Step {self.step_num} truncated: {truncated}")

        return observations, rewards, terminated, truncated, {agent: {} for agent in self.agents}

    def _validate_config(self, config):
        """Validate input configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If required parameters are missing or have invalid types
        """
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
        """Set basic attributes from configuration.

        Args:
            config: Configuration dictionary containing environment parameters
        """
        self.generalize = None
        self.truncateds = None
        self.terminateds = None
        self.resetted = None
        self.cur_param = None
        self.continue_steps_enable = None
        self.dynamic_queue = None
        self.sim_output = None
        self.init_method = None
        self.predefined_init_param = None
        self.reward_func = None
        self.max_step = None
        self.log_level = None
        self.device_mask_config = None
        self.generalize_specs_config = None
        self.dc_sim_config = None
        self.agent_assign_config = None
        self.param_range_config = None
        self.norm_specs_file = None
        self.sim_config = None
        self.netlist_folder_name = None
        self.specs_folder_name = None
        self.config_folder_name = None
        self.run_folder_name = None
        self.corner_sim = None
        self.dc_check = None
        self.region_extract = None

        for key in config:
            setattr(self, key, config[key])

        # Set flag attributes
        flag_mappings = {
            'continue_steps_enable': 'continue_steps_enable'
        }
        for dest, src in flag_mappings.items():
            setattr(self, dest, getattr(self, src, False))

    def _setup_paths(self):
        """Configure all environment paths."""
        self.current_path = os.getcwd()
        self.home_dir = os.path.expanduser("~")

        # Main working directory
        self.run_root_dir = os.path.join(
            self.home_dir,
            "AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env",
            self.run_folder_name
        )
        os.makedirs(self.run_root_dir, exist_ok=True)

        # Configuration file paths
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
        """Load all YAML configuration files."""
        self.sim_config_dict = self._load_yaml_config(self.sim_config)
        self.norm_specs = self._load_yaml_config(self.norm_specs_file)
        self.param_range_dict = self._load_yaml_config(self.param_range_config)
        self.agent_assign_dict = self._load_yaml_config(self.agent_assign_config)
        self.device_mask_dict = self._load_yaml_config(self.device_mask_config)
        self.dc_sim_config_dict = self._load_yaml_config(self.dc_sim_config)
        self.generalize_specs_config_dict = self._load_yaml_config(self.generalize_specs_config)

    def _load_yaml_config(self, file_path: str) -> dict:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            dict: Loaded configuration

        Raises:
            Exception: If file loading fails
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config {file_path}: {e}")
            raise

    def _setup_logging(self):
        """Configure logging system with specified level."""
        # Convert log level to logging constant
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s/%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)

        # Configure module-specific logger
        self.logger = logging.getLogger("SpiceEnv")
        self.logger.setLevel(log_level)
        self.logger.propagate = True

        # Validate logging configuration
        self._validate_log_config(root_logger)

    def _validate_log_config(self, root_logger):
        """Validate logging configuration by sending test messages.

        Args:
            root_logger: Configured root logger instance
        """
        test_messages = {
            logging.DEBUG: "Debug test message (should show if level <= DEBUG)",
            logging.INFO: "Info test message (should show if level <= INFO)",
            logging.WARNING: "Warning test message (should show if level <= WARNING)",
            logging.ERROR: "Error test message (should show if level <= ERROR)",
        }

        current_level = logging.getLevelName(root_logger.getEffectiveLevel())
        print(f"\nCurrent effective log level: {current_level}")

        for level, msg in test_messages.items():
            root_logger.log(level, msg)

        # Test external module logging
        external_logger = logging.getLogger("ExternalModule")
        external_logger.debug("External debug test")
        external_logger.info("External info test")

    def _initialize_spaces(self):
        """Initialize observation and action spaces."""
        # Generate parameter space
        self.param_space = copy.deepcopy(gen_param_space(self.param_range_dict))

        # Generate observation space
        obs_generator = gen_obs_space_w_region if self.region_extract else gen_obs_space
        flatten_obs = flatten_obs_space_w_region if self.region_extract else flatten_obs_space
        self.observation_space = flatten_obs(obs_generator(
            self.sim_config_dict,
            self.param_range_dict,
            self.agent_assign_dict
        ))

        # Generate masked action space
        self.action_space = gen_masked_continuous_action_space(
            self.device_mask_dict,
            self.agent_assign_dict
        )

    def _initialize_state_variables(self):
        """Initialize environment state variables."""
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

        # Reward continuation tracking
        self.continue_steps = 4  # Number of steps to continue after positive reward
        self.steps_after_positive_reward = 0
        self.had_positive_reward = False

        # Operation region tracking
        self.operation_region_dict_zero = {
            comp: 0 for comp in self.param_range_dict
            if comp != 'other_variable'
        }

        # Agent management
        self.possible_agents = list(self.agent_assign_dict.keys())
        self.agents = self.possible_agents
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

    def _import_reward_function(self):
        """Dynamically import reward calculation function."""
        try:
            reward_module = importlib.import_module('util.cal_reward')
            self.cal_reward = getattr(reward_module, self.reward_func)
        except (ImportError, AttributeError) as e:
            logging.error(f"Error importing reward function '{self.reward_func}': {e}")
            raise

    def _initialize_reset_variables(self):
        """Initialize all reset-related variables."""
        # State control variables
        self.steps_after_positive_reward = 0
        self.had_positive_reward = False
        self.step_num = 0
        self.resetted = False

        # Simulation result cache
        self.zero_sim_result = {}
        for sim in self.generalize_specs_config_dict:
            specs_tmp_dict = {}
            for specs_item in self.generalize_specs_config_dict[sim]:
                obj_type = self.generalize_specs_config_dict[sim][specs_item]['objective']
                specs_tmp_dict[specs_item] = 0.0 if obj_type == 'max' else 100.0
            self.zero_sim_result[sim] = specs_tmp_dict

        self.logger.info(f"Zero sim result: {self.zero_sim_result}")

        # Initialize parameters and specs
        self.cur_param = None
        self.ideal_specs = None
        self.norm_ideal_specs = None

    def _load_ideal_specs(self):
        """Load and normalize ideal specifications."""
        self.ideal_specs = generalize_config(self.generalize, self.ideal_specs_path)
        self.norm_ideal_specs = norm_ideal_spec(self.ideal_specs, self.norm_specs)

        self.logger.info(f"Ideal specs: {self.ideal_specs}")
        self.logger.info(f"Normalized ideal specs: {self.norm_ideal_specs}")

    def _generate_initial_parameters(self) -> dict:
        """Generate initial design parameters.

        Returns:
            dict: Initial parameter values
        """
        init_param = gen_init_param(
            self.init_method,
            self.predefined_init_param,
            True,
            self.device_mask_dict,
            self.param_space
        )
        return init_param

    def _setup_reset_environment(self, init_param: dict) -> str:
        """Set up reset environment and return working directory.

        Args:
            init_param: Initial parameter values

        Returns:
            str: Path to created working directory
        """
        working_dir = create_work_dir(self.run_root_dir, 'init')
        update_netlist(working_dir, self.sim_config_dict, init_param, self.unassigned_netlist_dir)
        return working_dir

    def _run_initial_dc_simulation(self, init_param: dict) -> Union[dict, None]:
        """Run initial DC simulation for operation region check.

        Args:
            init_param: Initial parameter values

        Returns:
            dict: Operation region information or None if disabled
        """
        if not self.region_extract:
            return None

        operation_region_dict = self._run_region_simulation_check('init_dc', init_param)
        self.logger.debug(f"DC simulation region results: {operation_region_dict}")
        return operation_region_dict

    def _run_region_simulation_check(self, dir_suffix: str, param: dict) -> dict:
        """Run region simulation with error handling.

        Args:
            dir_suffix: Directory suffix for simulation
            param: Current parameter values

        Returns:
            dict: Operation region results or zero values on failure
        """
        try:
            working_dir = create_work_dir(self.run_root_dir, dir_suffix)
            update_netlist(working_dir, self.dc_sim_config_dict, param, self.unassigned_netlist_dir)
            operation_region_dict = copy.deepcopy(
                run_region_simulation(working_dir, self.dc_sim_config_dict, self.sim_output))
            self.logger.debug(f"Operation region dict: {operation_region_dict}")

            # Validate result length
            if len(operation_region_dict) != len(self.operation_region_dict_zero):
                logging.warning(
                    f"Operation region dict length mismatch. Expected {len(self.operation_region_dict_zero)}, Got {len(operation_region_dict)}")
                return copy.deepcopy(self.operation_region_dict_zero)

            return operation_region_dict
        except Exception as e:
            logging.warning(f"Region simulation failed: {str(e)}")
            return copy.deepcopy(self.operation_region_dict_zero)

    def _save_reset_data(self, working_dir: str, param: dict, sim_result: dict, reward: float):
        """Save reset step data to file.

        Args:
            working_dir: Simulation working directory
            param: Parameter values used
            sim_result: Simulation results
            reward: Calculated reward
        """
        step_data = {
            'param': param,
            'sim_result': sim_result,
            'reward': reward,
            'corner': 'tt'
        }
        with open(os.path.join(working_dir, 'result.pkl'), 'wb') as f:
            pickle.dump(step_data, f)

    def _process_simulation_result(self, sim_result: dict, param: dict,
                                   operation_region_dict: dict = None) -> tuple:
        """Process simulation results into observations and reward.

        Args:
            sim_result: Raw simulation results
            param: Current parameter values
            operation_region_dict: Operation region information

        Returns:
            tuple: (observations, sim_result, reward)
        """
        # Result normalization
        norm_sim_result = norm_sim_spec(sim_result, self.norm_specs)

        self.logger.info(f"Simulation result: {sim_result}")
        self.logger.debug(f"Normalized result: {norm_sim_result}")

        # Generate observation space
        if self.region_extract:
            observation_detail = copy.deepcopy(
                update_obs_space_w_region(
                    self.norm_ideal_specs,
                    norm_sim_result,
                    param,
                    operation_region_dict
                )
            )
            observation = copy.deepcopy(flatten_observation_w_region(observation_detail))
        else:
            observation_detail = copy.deepcopy(
                update_obs_space(
                    self.norm_ideal_specs,
                    norm_sim_result,
                    param
                )
            )
            observation = copy.deepcopy(flatten_observation(observation_detail))

        self.logger.debug(f"Observation detail: {observation_detail}")
        self.logger.debug(f"Flattened observation: {observation}")

        # Build multi-agent observations
        observations = {agent: observation for agent in self.agents}

        # Calculate reward
        rew = self.cal_reward(self.ideal_specs, sim_result, self.norm_specs)

        return observations, sim_result, rew

    def _run_simulation(self, working_dir: str, param: dict,
                        operation_region_dict: dict = None) -> tuple:
        """Execute simulation with retry mechanism.

        Args:
            working_dir: Simulation working directory
            param: Current parameter values
            operation_region_dict: Operation region information

        Returns:
            tuple: Processed simulation results
        """

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

    def _convert_actions_to_params(self, action_dict: dict) -> tuple:
        """Convert agent actions to design parameters.

        Args:
            action_dict: Dictionary of agent actions

        Returns:
            tuple: (updated_param, all_action_flatten)
        """
        # Apply device masking
        if self.device_mask_dict:
            all_action_flatten = masked_action_dict_mapping(self.device_mask_dict, action_dict)
            self.logger.debug(f"Action dict w/ device mask: {all_action_flatten}")
        else:
            all_action_flatten = OrderedDict()
            for group in action_dict.values():
                all_action_flatten.update(group)
            self.logger.debug(f"Action dict w/o device mask: {all_action_flatten}")

        # Convert to parameters
        updated_param = action2param(
            self.action_mask,
            self.device_mask_dict,
            all_action_flatten,
            self.param_range_dict
        )

        return updated_param, all_action_flatten

    def _perform_step_dc_check(self, param: dict) -> tuple:
        """Perform DC operating region check.

        Args:
            param: Current parameter values

        Returns:
            tuple: (region_info, validity_flag)
        """
        if not self.region_extract:
            return None, True

        operation_region_dict = self._run_region_simulation_check('dc', param)
        valid_regions = [1, 2, 3]  # triode/saturation/sub-threshold
        valid_param = all(v in valid_regions for v in operation_region_dict.values())
        if valid_param:
            self.logger.debug(f"DC check passed. Operation region dict: {operation_region_dict}")
        else:
            self.logger.warning(f"DC check failed. Operation region dict: {operation_region_dict}")

        return operation_region_dict, valid_param

    def _handle_dc_check_failure(self, param: dict, region_info: dict) -> tuple:
        """Handle DC check failure scenario.

        Args:
            param: Current parameter values
            region_info: Operation region information

        Returns:
            tuple: (observations, reward)
        """
        self.logger.info(f"Step {self.step_num}: DC check failed with region extraction enabled")

        sim_result = copy.deepcopy(self.zero_sim_result)
        norm_result = norm_sim_spec(sim_result, self.norm_specs)

        # Generate observations
        if self.region_extract:
            obs_detail = update_obs_space_w_region(
                self.norm_ideal_specs,
                norm_result,
                param,
                region_info
            )
            observation = flatten_observation_w_region(obs_detail)
        else:
            obs_detail = update_obs_space(
                self.norm_ideal_specs,
                norm_result,
                param
            )
            observation = flatten_observation(obs_detail)

        self.logger.info(f"DC check failed. Assigning penalty reward: -10")
        return {agent: observation for agent in self.agents}, -10

    def _run_main_simulation(self, param: dict, region_info: dict) -> tuple:
        """Execute main simulation process.

        Args:
            param: Current parameter values
            region_info: Operation region information

        Returns:
            tuple: (observations, sim_result, reward, working_dir)
        """
        main_dir = create_work_dir(self.run_root_dir, 'tt')
        update_netlist(main_dir, self.sim_config_dict, param, self.unassigned_netlist_dir)
        observations, sim_result, reward = self._run_simulation(main_dir, param, region_info)
        self.logger.info(f"TT simulation result: {sim_result} with reward: {reward}")
        return observations, sim_result, reward, main_dir

    def _save_step_data(self, working_dir: str, param: dict,
                        sim_result: dict, reward: float, corner: str = 'tt'):
        """Save step simulation data to file.

        Args:
            working_dir: Simulation working directory
            param: Parameter values used
            sim_result: Simulation results
            reward: Calculated reward
            corner: Process corner identifier
        """
        step_data = {
            'param': param,
            'sim_result': sim_result,
            'reward': reward,
            'corner': corner
        }
        with open(os.path.join(working_dir, 'result.pkl'), 'wb') as f:
            pickle.dump(step_data, f)

    def _process_corner_results(self, main_dir: str, param: dict,
                                region_info: dict, tt_result: dict) -> float:
        """Process corner simulation results and calculate unified reward.

        Args:
            main_dir: Main simulation directory
            param: Current parameter values
            region_info: Operation region information
            tt_result: TT corner simulation results

        Returns:
            float: Unified reward value across all corners
        """
        # 1. Aggregate all corner results
        corner_results = self._handle_corner_simulations(main_dir, param, region_info)
        corner_results['tt'] = {
            'working_dir': main_dir,
            'sim_result': tt_result['sim_result'],
            'reward': tt_result['reward']
        }

        # 2. Calculate unified reward
        min_reward = min(v['reward'] for v in corner_results.values())
        final_reward = max(min_reward, 10) if min_reward < 0 else min_reward

        # 3. Save all results
        for corner, data in corner_results.items():
            self._save_step_data(
                data['working_dir'],
                param,
                data['sim_result'],
                final_reward,
                corner
            )
        self.logger.info("Processed corner simulations")
        self.logger.debug(f"Corner results: {corner_results}")

        return final_reward

    def _assign_rewards(self, base_reward: float) -> dict:
        """Distribute base reward to all agents.

        Args:
            base_reward: Calculated base reward value

        Returns:
            dict: Agent-specific reward dictionary
        """
        return {agent: base_reward for agent in self.agents}

    def _determine_termination(self, reward: float) -> tuple:
        """Determine termination conditions for all agents.

        Args:
            reward: Current step reward value

        Returns:
            tuple: (terminated_dict, truncated_dict)
        """
        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}

        # Continuation step logic
        if self.continue_steps_enable:
            if reward > 0 and not self.had_positive_reward:
                self.had_positive_reward = True
                self.steps_after_positive_reward = 0
            elif self.had_positive_reward:
                self.steps_after_positive_reward += 1

            episode_over = (
                    (self.had_positive_reward and
                     (self.steps_after_positive_reward >= self.continue_steps or
                      self.step_num >= self.max_step)) or
                    (self.step_num >= self.max_step)
            )

            if episode_over:
                for agent in self.agents:
                    if self.had_positive_reward:
                        terminated[agent] = True
                        self.terminateds.add(agent)
                    else:
                        truncated[agent] = True
                        self.truncateds.add(agent)
        else:
            # Basic termination logic
            if reward > 0:
                for agent in self.agents:
                    terminated[agent] = True
                    self.terminateds.add(agent)
            if self.step_num >= self.max_step:
                for agent in self.agents:
                    truncated[agent] = True
                    self.truncateds.add(agent)

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        return terminated, truncated

    def _handle_corner_simulations(self, main_dir: str, param: dict,
                                   operation_region_dict: dict) -> dict:
        """Handle multi-corner simulation execution and results collection.

        Args:
            main_dir: Main simulation directory path
            param: Current design parameters
            operation_region_dict: Device operation region information

        Returns:
            dict: Aggregated corner simulation results
        """
        corner_results = {}
        for corner in ['ff', 'fs', 'sf', 'ss']:
            # Create corner-specific directory
            corner_dir = create_corner_work_dir(main_dir, corner)

            # Update netlist with corner parameters
            update_netlist(corner_dir, self.sim_config_dict, param,
                           self.unassigned_netlist_dir, corner)

            # Execute simulation
            obs, sim_result, rew = self._run_simulation(
                corner_dir,
                param,
                operation_region_dict
            )

            # Store results
            corner_results[corner] = {
                'working_dir': corner_dir,
                'sim_result': sim_result,
                'reward': rew
            }
            self.logger.info(f"Corner {corner} simulation result: {sim_result} with reward: {rew}")

        return corner_results
