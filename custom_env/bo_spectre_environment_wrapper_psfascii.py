import os
import yaml
import numpy as np
import logging
import copy
from typing import Dict, Tuple, List, Any, Optional
import pickle
from datetime import datetime
import importlib

from bo_spice_single_new_psfascii import RllibSingleAnalogDesignAutoEnv
from util.action2param import action2param
from util.device_mask_bo import masked_action_dict_mapping
from collections import OrderedDict

# Import reward calculation functions from cal_reward
try:
    cal_reward = importlib.import_module('util.cal_reward')
except ImportError:
    try:
        cal_reward = importlib.import_module('cal_reward')
    except ImportError:
        logging.error("Could not import cal_reward module. Make sure it's in the PYTHONPATH.")
        raise

class SpectreEnvironmentWrapper:
    """
    Wrapper class for the Spectre environment to be used with Bayesian Optimization
    Provides an interface compatible with the circuit_evaluator.py
    """
    
    def __init__(self, 
                 config_folder: str,
                 run_folder: str,
                 reward_func: str = "cal_reward_Haoqiang",
                 device_mask_flag: bool = True,
                 batch_size: int = 8,
                 env_config: Dict[str, Any] = None):
        """
        Initialize the Spectre environment wrapper
        
        Args:
            config_folder: Path to configuration folder
            run_folder: Path to simulation run folder
            reward_func: Name of reward function to use from cal_reward
            device_mask_flag: Enable device matching constraints
            batch_size: Size of parallel evaluation batches
            env_config: Optional pre-configured environment settings
        """
        self.config_folder = config_folder
        self.run_folder = run_folder
        self.device_mask_flag = device_mask_flag
        self.batch_size = batch_size
        
        # Extract folder names from config_folder path
        folder_parts = config_folder.split('/')
        config_folder_name = folder_parts[-1] if folder_parts else "config_Haoqiang_Regroup_single"
        
        # Parse appropriate folder names from config_folder
        config_name_parts = config_folder_name.split('_')
        circuit_type = config_name_parts[1] if len(config_name_parts) > 1 else "Haoqiang"
        
        # Initialize environment config
        if env_config is None:
            self.env_config = {
                "generalize": True,
                "max_step": 1,  # Only single step needed for BO
                "netlist_folder_name": f"netlist_template_{circuit_type}",
                "specs_folder_name": f"sampled_specs_{circuit_type}",
                "config_folder_name": config_folder_name,
                "run_folder_name": run_folder,
                "sim_output": False,
                "init_method": "random",
                "corner_sim": False,
                "dc_check": False,
                "region_extract": False,
                "dynamic_queue": False,
                "log_level": "INFO",
                "reward_func": reward_func,
                "continue_steps_enable": False
            }
        else:
            self.env_config = env_config
            # Ensure run_folder is set correctly
            self.env_config["run_folder_name"] = run_folder
        
        # Get reward function from cal_reward module
        if hasattr(cal_reward, reward_func):
            self.reward_func = getattr(cal_reward, reward_func)
            logging.info(f"Using reward function {reward_func} from cal_reward module")
        else:
            available_funcs = [name for name in dir(cal_reward) 
                              if name.startswith('cal_reward_') and callable(getattr(cal_reward, name))]
            logging.warning(f"Reward function {reward_func} not found in cal_reward module")
            logging.warning(f"Available reward functions: {available_funcs}")
            
            # Default to a function that exists
            fallback_func = next(iter([name for name in available_funcs]), None)
            if fallback_func:
                self.reward_func = getattr(cal_reward, fallback_func)
                logging.warning(f"Falling back to {fallback_func}")
            else:
                raise ValueError(f"No reward functions found in cal_reward module")
        
        # Initialize environment
        self.env = RllibSingleAnalogDesignAutoEnv(self.env_config)
        
        # Reset environment to initialize configuration
        self.env.reset()
        
        # Store important references
        self.param_range_dict = self.env.param_range_dict
        self.device_mask_dict = self.env.device_mask_dict
        self.ideal_specs = self.env.ideal_specs
        self.norm_specs = self.env.norm_specs
        
        # Initialize parameter tracking
        self.current_params = None
        
        logging.info(f"SpectreEnvironmentWrapper initialized with config: {self.env_config}")
    
    def evaluate_single(self, params: np.ndarray) -> float:
        """
        Evaluate a single parameter combination
        
        Args:
            params: Normalized parameter values from optimizer
            
        Returns:
            reward: Evaluation reward value
        """
        logging.info("=" * 50)
        logging.info("Starting new evaluation")
        logging.info(f"Parameters from optimizer: {params}")
        
        try:
            # Convert normalized params to actions
            actions = self._convert_params_to_actions(params)
            
            # Reset environment to get a clean state
            self.env.reset()
            
            # Step environment with actions
            observation, reward, terminated, truncated, info = self.env.step(actions)
            
            # Save this evaluation result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_dir = os.path.join(self.run_folder, f"eval_{timestamp}")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Save parameters and results
            # Ensure params are properly converted to a serializable format
            params_list = params.tolist() if isinstance(params, np.ndarray) else params
            
            result_data = {
                'params': params_list,
                'reward': reward,
                'observation': observation,
                'terminated': terminated,
                'truncated': truncated
            }
            
            with open(os.path.join(eval_dir, 'evaluation.pkl'), 'wb') as f:
                pickle.dump(result_data, f)
            
            logging.info(f"Evaluation completed with reward: {reward}")
            logging.info(f"Results saved to {eval_dir}")
            
            return reward
            
        except Exception as e:
            logging.error(f"Error in evaluation: {str(e)}")
            logging.exception("Exception details:")
            return -np.inf
            
    def evaluate_batch(self, params_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate multiple parameter combinations sequentially
        
        Args:
            params_batch: Batch of normalized parameter values
            
        Returns:
            rewards: Array of evaluation rewards
        """
        rewards = []
        for params in params_batch:
            reward = self.evaluate_single(params)
            rewards.append(reward)
        return np.array(rewards)
    
    def save_best_parameters(self, params: np.ndarray, reward: float, 
                            save_path: Optional[str] = None) -> None:
        """
        Save the best parameters found during optimization
        
        Args:
            params: Best parameters from optimizer
            reward: Best reward achieved
            save_path: Optional path to save results
        """
        if save_path is None:
            save_path = os.path.join(self.run_folder, "best_parameters")
            os.makedirs(save_path, exist_ok=True)
        
        os.makedirs(save_path, exist_ok=True)
        # Convert normalized params to circuit parameters
        actions = self._convert_params_to_actions(params)
        
        # Reset and step to get final state
        self.env.reset()
        observation, final_reward, terminated, truncated, info = self.env.step(actions)
        
        # Save parameters and results
        result_data = {
            'normalized_params': params,
            'actions': actions,
            'reward': reward,
            'final_reward': final_reward,
            'observation': observation
        }
        
        with open(os.path.join(save_path, 'best_parameters.pkl'), 'wb') as f:
            pickle.dump(result_data, f)
            
        # Save parameters in YAML format for readability
        param_dict = {
            'normalized_params': params.tolist(),
            'reward': float(reward)
        }
        
        with open(os.path.join(save_path, 'best_parameters.yaml'), 'w') as f:
            yaml.dump(param_dict, f, default_flow_style=False)
    
    # def _convert_params_to_actions(self, params: np.ndarray) -> dict:
    #     """
    #     Convert normalized parameters to environment actions
        
    #     This is a critical function that transforms the continuous parameter space from 
    #     the Bayesian Optimizer into the action space expected by the spice_single_new environment.
        
    #     For spice_single_new.py, we need to create an action dictionary that looks like:
    #     {'single_agent': params_array}
        
    #     Args:
    #         params: Normalized parameter values from optimizer
            
    #     Returns:
    #         action_dict: Action dictionary for environment
    #     """
    #     # Get parameter names
    #     param_names = self._get_param_names()
        
    #     # Check dimensions
    #     if len(params) != len(param_names):
    #         logging.warning(f"Parameter length mismatch: {len(params)} vs {len(param_names)}")
    #         # Pad or truncate as necessary
    #         if len(params) < len(param_names):
    #             params = np.pad(params, (0, len(param_names) - len(params)), 'constant', constant_values=0.5)
    #         else:
    #             params = params[:len(param_names)]
        
    #     # Make sure we have a NumPy array
    #     if not isinstance(params, np.ndarray):
    #         params = np.array(params)
        
    #     # For spice_single_new.py, we need to create an action dictionary
    #     # with 'single_agent' as the key
    #     action_dict = {'single_agent': params}
        
    #     return action_dict
    def _convert_params_to_actions(self, params: np.ndarray) -> dict:
        """
        将归一化参数转换为环境动作字典的正确格式，
        即形如 {'C0': array([...]), 'C1': array([...]), ..., 'ibias': array([...])}
        """
        # 从 param_range_dict 中提取参数名和各参数维度（如果存在 dim 属性，否则默认为1）
        action_dict = {}
        index = 0
        # 假设 self.param_range_dict 的结构如下：
        # {
        #     "some_key": {
        #         "params": [
        #             {"variable_name": "C0", "dim": 1},
        #             {"variable_name": "M0", "dim": 3},
        #             ...
        #         ]
        #     },
        #     ...
        # }
        for param_group in self.param_range_dict.values():
            for p in param_group['params']:
                var_name = p['variable_name']
                dim = p.get('dim', 1)  # 默认维度1
                # 检查参数数量是否足够，否则警告或者补0处理
                if index + dim > len(params):
                    
                    slice_vals = np.full((dim,), 0.5)  # 默认填充值 0.5
                else:
                    slice_vals = params[index: index + dim]
                # 保证转换为 NumPy 数组（若 slice_vals 只有一个元素，这里也转换为 array）
                action_dict[var_name] = np.array(slice_vals)
                index += dim
        return action_dict
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get parameter bounds
        
        Returns:
            lower_bounds, upper_bounds: Parameter bounds arrays
        """
        # Bounds for normalized parameters are always 0 to 1
        dim = self.get_param_dims()
        lower_bounds = np.zeros(dim)
        upper_bounds = np.ones(dim)
        return lower_bounds, upper_bounds
    
    def get_param_dims(self) -> int:
        """
        Get parameter space dimensionality
        
        Returns:
            dims: Number of parameters
        """
        return len(self._get_param_names())
    
    def _get_param_names(self) -> List[str]:
        """
        Get list of parameter names
        
        Returns:
            list: Parameter names
        """
        # Extract parameter names from param_range_dict
        # This is more reliable than using action space keys
        names = []
        for param in self.param_range_dict.values():
            for p in param['params']:
                names.append(p['variable_name'])
        return names