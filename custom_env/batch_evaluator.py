from multiprocessing import Pool
from tqdm import tqdm
import time
from typing import Dict, List, Any, Tuple
from collections import OrderedDict
import pandas as pd
import os
import yaml
import logging
import copy

from util.assign_param2netlist import update_netlist
from util.util_func import retry_decorator, create_work_dir
from util.run_spectre_simulation import run_dynamic_simulation_psfascii


def _parallel_evaluation_task(task_input: Tuple[str, str, Dict, Dict, Dict, str]) -> Tuple[str, Dict]:
    """
    Wrapper function for parallel evaluation tasks.

    Args:
        task_input: Tuple containing:
            - base_folder: Base directory for results
            - unassigned_netlist_dir: Directory for netlist templates
            - param_dict: Circuit parameters
            - sim_config: Simulation configuration
            - zero_result: Default zero result dictionary
            - corner: Corner for evaluation

    Returns:
        Tuple containing corner tag and simulation results
    """
    base_folder, unassigned_netlist_dir, param_dict, sim_config, zero_result, corner = task_input

    # Create unique working directory for this task
    working_dir = create_work_dir(base_folder)
    working_dir = f"{working_dir}_{corner}"

    try:
        sim_result = single_evaluation(
            working_dir=working_dir,
            unassigned_netlist_dir=unassigned_netlist_dir,
            updated_param=param_dict,
            sim_config_dict=sim_config,
            zero_sim_result=zero_result,
            corner_tag=corner
        )
        return corner, sim_result
    except Exception as e:
        logging.error(f"Evaluation failed for corner {corner}: {str(e)}")
        return corner, zero_result

def single_evaluation(working_dir, unassigned_netlist_dir, updated_param, sim_config_dict, zero_sim_result, corner_tag):

    sim_result = None
    sim_output_enable_tag = False
    dynamic_queue_tag = False

    os.makedirs(working_dir, exist_ok=True)
    print(f"Working_dir: {working_dir}")

    update_netlist(working_dir, sim_config_dict, updated_param, unassigned_netlist_dir, corner_tag)
    print(f"Netlist updated for corner: {corner_tag}")

    @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=zero_sim_result)
    def _run_simulation_with_retry():
        print(f"Running simulation for corner: {corner_tag}")
        return run_dynamic_simulation_psfascii(working_dir, sim_config_dict, zero_sim_result,
                                               sim_output_enable_tag, dynamic_queue_tag)

    try:
        sim_result = copy.deepcopy(_run_simulation_with_retry())
    except Exception as e:
        logging.warning(f"Step Warning!!!: {e}. sim_result is {sim_result}."
                        f" Simulation failed, use zero result instead.")
        sim_result = copy.deepcopy(zero_sim_result)

    print(f"Step!!!Simulation result: {sim_result} in corner: {corner_tag}")

    return sim_result


def batch_evaluation_parallel(
        base_folder: str,
        unassigned_netlist_dir: str,
        generalize_specs_config_dict: Dict[str, Dict[str, Any]],
        init_param_dict: List[Dict[str, Any]],
        sim_config_dict: Dict[str, Any],
        corner_list: List[str],
        num_workers: int = 4) -> None:
    """
    Parallel batch evaluation function for multiple parameter sets across different corners.

    Args:
        base_folder: Base directory for simulation results
        unassigned_netlist_dir: Directory containing unassigned netlist templates
        generalize_specs_config_dict: Generalized specs configuration dictionary
        init_param_dict: List of parameter dictionaries to evaluate
        sim_config_dict: Simulation configuration dictionary
        corner_list: List of corners to evaluate
        num_workers: Number of parallel workers (default: 4)

    Returns:
        None (Creates an Excel file with evaluation results)
    """
    # Initialize results dictionary
    evaluation_dict = OrderedDict()

    # Generate zero result dictionary
    zero_sim_result = {}
    for sim in generalize_specs_config_dict:
        specs_tmp_dict = {}
        for specs_item in generalize_specs_config_dict[sim]:
            if generalize_specs_config_dict[sim][specs_item]['objective'] == 'max':
                specs_tmp_dict[specs_item] = 0.0
            if generalize_specs_config_dict[sim][specs_item]['objective'] == 'min':
                specs_tmp_dict[specs_item] = 100.0
            if generalize_specs_config_dict[sim][specs_item]['objective'] == 'range':
                specs_tmp_dict[specs_item] = 100.0
        zero_sim_result[sim] = specs_tmp_dict

    print(f"Starting parallel evaluation with {num_workers} workers")

    # Create process pool
    with Pool(processes=num_workers) as pool:
        # Process each parameter set
        for param_idx, param_dict in enumerate(init_param_dict, 1):
            param_key = f"param_set_{param_idx}"
            evaluation_dict[param_key] = {}

            # Prepare tasks for all corners
            tasks = [
                (base_folder, unassigned_netlist_dir, param_dict, sim_config_dict,
                 zero_sim_result, corner) for corner in corner_list
            ]

            # Run parallel evaluation with progress bar
            results = []
            with tqdm(total=len(tasks), desc=f"Evaluating {param_key}") as pbar:
                for result in pool.imap_unordered(_parallel_evaluation_task, tasks):
                    results.append(result)
                    pbar.update()

            # Store results
            for corner, sim_result in results:
                evaluation_dict[param_key][corner] = {
                    "param": param_dict,
                    "result": sim_result
                }

    # Create Excel report
    _create_excel_report(evaluation_dict, base_folder)


def _create_excel_report(evaluation_dict: Dict[str, Any], base_folder: str) -> None:
    """
    Create Excel report from evaluation dictionary with separate sheets for each parameter set.

    Args:
        evaluation_dict: Dictionary containing evaluation results
        base_folder: Base directory to save the Excel file

    Returns:
        None
    """
    # Group data by param_set
    param_set_data = {}

    # Process each parameter set and corner
    for param_set, corner_data in evaluation_dict.items():
        param_rows = []
        for corner, data in corner_data.items():
            row = {
                'param_set': param_set,
                'corner': corner
            }

            # Add results
            for sim_type, sim_metrics in data['result'].items():
                for metric_name, metric_value in sim_metrics.items():
                    row[f"result_{metric_name}"] = metric_value

            # Add parameters
            for param_key, param_value in data['param'].items():
                row[f"param_{param_key}"] = param_value

            param_rows.append(row)

        param_set_data[param_set] = param_rows

    # Create Excel file with separate sheets
    output_file = os.path.join(base_folder, 'evaluation_results.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Create individual sheets for each param_set
        for param_set, rows in param_set_data.items():
            df = pd.DataFrame(rows)
            # Reorder columns
            cols = df.columns.tolist()
            cols.remove('param_set')
            cols = ['param_set'] + cols
            df = df[cols]
            # Write to sheet
            df.to_excel(writer, sheet_name=param_set, index=False)

    print(f"Evaluation results saved to {output_file} with separate sheets for each parameter set")

def validate_configurations(base_folder: str,
                            unassigned_netlist_dir: str,
                            generalize_specs_config_dict: Dict,
                            init_param_dict: List[Dict],
                            sim_config_dict: Dict) -> bool:
    """
    Validate all input configurations before starting evaluation.

    Args:
        base_folder: Base directory for results
        unassigned_netlist_dir: Directory for netlist templates
        generalize_specs_config_dict: Specs configuration
        init_param_dict: List of parameter sets
        sim_config_dict: Simulation configuration

    Returns:
        bool: True if all validations pass

    Raises:
        ValueError: If any validation fails
    """
    # Check directories exist
    if not os.path.exists(base_folder):
        try:
            os.makedirs(base_folder)
        except Exception as e:
            raise ValueError(f"Cannot create base folder: {str(e)}")

    if not os.path.exists(unassigned_netlist_dir):
        raise ValueError(f"Netlist template directory does not exist: {unassigned_netlist_dir}")

    # Validate configurations
    if not generalize_specs_config_dict:
        raise ValueError("Empty generalize specs configuration")

    if not init_param_dict:
        raise ValueError("Empty parameter sets")

    if not sim_config_dict:
        raise ValueError("Empty simulation configuration")

    # Check parameter sets format
    for idx, param_set in enumerate(init_param_dict):
        if not isinstance(param_set, dict):
            raise ValueError(f"Invalid parameter set format at index {idx}")

    return True


if __name__ == "__main__":
    # Configuration paths
    config_base = '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/config/config_AXS_Simple/'
    paths = {
        'base_folder': '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/run_test/',
        'unassigned_netlist_dir': '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_AXS_Corner/',
        'generalize_specs_config': os.path.join(config_base, 'generalize_specs.yaml'),
        'init_param': os.path.join(config_base, 'init_batch.yaml'),
        'sim_config': os.path.join(config_base, 'simulation.yaml')
    }

    # Load configurations
    try:
        with open(paths['generalize_specs_config'], 'r') as f:
            generalize_specs_config_dict = yaml.safe_load(f)
        with open(paths['init_param'], 'r') as f:
            init_param_dict = yaml.safe_load(f)
        with open(paths['sim_config'], 'r') as f:
            sim_config_dict = yaml.safe_load(f)

        # Validate configurations
        validate_configurations(
            paths['base_folder'],
            paths['unassigned_netlist_dir'],
            generalize_specs_config_dict,
            init_param_dict,
            sim_config_dict
        )

        # Start with tt corner
        corner_list = ['tt']
        corner_pattern_1 = ['fs', 'sf', 'ff', 'ss']
        cornet_pattern_2 = ['ss', 'ff']
        temp_pattern = ['b40', '125']
        ESR_pattern = ['p2', '1']
        VDDI_pattern = ['1p65', '1p95']
        # Generate a corner summary list: {corner_pattern_1}_{cornet_pattern_2}_{cornet_pattern_2}_{temp_pattern}
        corner_summary = [f"{corner1}_{corner2}_{corner3}_{temp}_{ESR}_{VDDI}" for corner1 in corner_pattern_1 for corner2 in
                          cornet_pattern_2 for corner3 in cornet_pattern_2 for temp in temp_pattern for ESR in ESR_pattern for VDDI in VDDI_pattern]
        corner_list.extend(corner_summary)
        print(f"Evaluation Corner List: {corner_list}")

        # Set number of parallel workers
        num_workers = 60  # Adjust based on system capabilities

        # Run parallel evaluation
        batch_evaluation_parallel(
            paths['base_folder'],
            paths['unassigned_netlist_dir'],
            generalize_specs_config_dict,
            init_param_dict,
            sim_config_dict,
            corner_list,
            num_workers
        )

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise