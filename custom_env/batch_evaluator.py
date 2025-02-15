import os
import yaml
import copy
import logging
from collections import OrderedDict
from typing import Dict, Any, List, Optional

from util.util_func import create_work_dir
from util.assign_param2netlist import update_netlist
from util.run_spectre_simulation import run_dynamic_simulation_psfascii, run_region_simulation_psfascii
from util.normlization import norm_sim_spec
from util.update_obs_space import update_obs_space_w_region, update_obs_space


class SimulationEvaluator:
    """
    A class to evaluate circuit parameters through simulation

    Attributes:
        config_folder (str): Path to configuration folder
        run_root_dir (str): Root directory for simulation runs
        netlist_dir (str): Directory containing netlist templates
        sim_output (bool): Flag to show simulation output
        region_extract (bool): Flag to extract operation region
        dc_check (bool): Flag to perform DC check
        dynamic_queue (bool): Flag for dynamic queue in simulation
    """

    def __init__(
            self,
            config_folder: str,
            run_root_dir: str,
            netlist_dir: str,
            sim_output: bool = False,
            region_extract: bool = False,
            dc_check: bool = False,
            dynamic_queue: bool = False
    ):
        # Load configurations
        self.config_folder = config_folder
        self.run_root_dir = os.path.expanduser(run_root_dir)
        self.netlist_dir = netlist_dir

        # Set flags
        self.sim_output = sim_output
        self.region_extract = region_extract
        self.dc_check = dc_check
        self.dynamic_queue = dynamic_queue

        # Load config files
        with open(os.path.join(config_folder, "simulation.yaml"), 'r') as f:
            self.sim_config = yaml.safe_load(f)
        with open(os.path.join(config_folder, "simulation_region.yaml"), 'r') as f:
            self.dc_sim_config = yaml.safe_load(f)
        with open(os.path.join(config_folder, "norm_specs.yaml"), 'r') as f:
            self.norm_specs = yaml.safe_load(f)

        # Initialize zero sim result template
        self.zero_sim_result = self._init_zero_sim_result()

        # Create run directory if not exists
        os.makedirs(self.run_root_dir, exist_ok=True)

    def _init_zero_sim_result(self) -> Dict:
        """Initialize zero simulation result template"""
        zero_result = {}
        with open(os.path.join(self.config_folder, "generalize_specs.yaml"), 'r') as f:
            specs_config = yaml.safe_load(f)

        for sim in specs_config:
            specs_tmp_dict = {}
            for specs_item in specs_config[sim]:
                if specs_config[sim][specs_item]['objective'] == 'max':
                    specs_tmp_dict[specs_item] = 0.0
                elif specs_config[sim][specs_item]['objective'] == 'min':
                    specs_tmp_dict[specs_item] = 100.0
                elif specs_config[sim][specs_item]['objective'] == 'range':
                    specs_tmp_dict[specs_item] = 100.0
            zero_result[sim] = specs_tmp_dict

        return zero_result

    def _evaluate_single_param(
            self,
            param_dict: Dict[str, Any],
            param_index: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single parameter set through simulation

        Args:
            param_dict: Dictionary containing circuit parameters
            param_index: Index of the parameter set

        Returns:
            Dictionary containing evaluation results
        """
        # Create working directory for this evaluation
        work_dir_path = create_work_dir(self.run_root_dir)
        work_dir = f"{work_dir_path}_{param_index}"
        os.makedirs(work_dir, exist_ok=True)

        operation_region_dict = None
        valid_param = True

        # Run DC check if enabled
        if self.region_extract:
            try:
                dc_work_dir = f"{work_dir_path}_{param_index}_dc"
                os.makedirs(dc_work_dir, exist_ok=True)

                update_netlist(
                    dc_work_dir,
                    self.dc_sim_config,
                    param_dict,
                    self.netlist_dir
                )

                operation_region_dict = copy.deepcopy(
                    run_region_simulation_psfascii(
                        dc_work_dir,
                        self.dc_sim_config,
                        self.sim_output
                    )
                )

                # Check operation regions
                operation_region_list = list(operation_region_dict.values())
                valid_param = all(item in [1, 2, 3] for item in operation_region_list)

            except Exception as e:
                logging.warning(f"DC check failed for param {param_index}: {str(e)}")
                valid_param = False
                operation_region_dict = {}

        # Run main simulation
        try:
            update_netlist(work_dir, self.sim_config, param_dict, self.netlist_dir)

            sim_result = copy.deepcopy(
                run_dynamic_simulation_psfascii(
                    work_dir,
                    self.sim_config,
                    self.zero_sim_result,
                    self.sim_output,
                    self.dynamic_queue
                )
            )
        except Exception as e:
            logging.warning(f"Simulation failed for param {param_index}: {str(e)}")
            sim_result = copy.deepcopy(self.zero_sim_result)

        return {
            'param_index': param_index,
            'parameters': param_dict,
            'simulation_result': sim_result,
            'work_dir': work_dir,
            'valid_param': valid_param,
            'operation_region': operation_region_dict
        }

    def evaluate_params_from_yaml(self, yaml_path: str) -> List[Dict[str, Any]]:
        """
        Evaluate multiple parameter sets from a YAML file

        Args:
            yaml_path: Path to YAML file containing parameter sets

        Returns:
            List of evaluation results for each parameter set
        """
        # Load parameters from YAML
        with open(yaml_path, 'r') as f:
            param_sets = yaml.safe_load(f)

        results = []
        for idx, params in enumerate(param_sets):
            try:
                result = self._evaluate_single_param(params, idx)
                results.append(result)
                logging.info(f"Evaluated parameter set {idx}")
            except Exception as e:
                logging.error(f"Failed to evaluate parameter set {idx}: {str(e)}")
                continue

        return results