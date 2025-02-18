from util.assign_param2netlist import update_netlist
from util.util_func import retry_decorator
from util.run_spectre_simulation import run_dynamic_simulation_psfascii
import copy
import os
import pandas as pd
from collections import OrderedDict
import logging
from typing import Dict, List, Any
from util.util_func import create_work_dir
import yaml

def single_evaluation(working_dir, unassigned_netlist_dir, updated_param, sim_config_dict, zero_sim_result, corner_tag):

    sim_result = None
    sim_output_enable_tag = False
    dynamic_queue_tag = False

    os.makedirs(working_dir, exist_ok=True)
    logging.info(f"Working_dir: {working_dir}")

    update_netlist(working_dir, sim_config_dict, updated_param, unassigned_netlist_dir, corner_tag)
    logging.info(f"Netlist updated for corner: {corner_tag}")

    @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=zero_sim_result)
    def _run_simulation_with_retry():
        logging.info(f"Running simulation for corner: {corner_tag}")
        return run_dynamic_simulation_psfascii(working_dir, sim_config_dict, zero_sim_result,
                                               sim_output_enable_tag, dynamic_queue_tag)

    try:
        sim_result = copy.deepcopy(_run_simulation_with_retry())
    except Exception as e:
        logging.warning(f"Step Warning!!!: {e}. sim_result is {sim_result}."
                        f" Simulation failed, use zero result instead.")
        sim_result = copy.deepcopy(zero_sim_result)

    logging.info(f"Step!!!Simulation result: {sim_result} in corner: {corner_tag}")

    return sim_result



def batch_evaluation(base_folder: str,
                     unassigned_netlist_dir: str,
                     generalize_specs_config_dict: Dict[str, Dict[str, Any]],
                     init_param_dict: List[Dict[str, Any]],
                     sim_config_dict: Dict[str, Any],
                     corner_list: List[str]) -> None:
    """
    Batch evaluation function for multiple parameter sets across different corners.

    Args:
        base_folder: Base directory for simulation results
        unassigned_netlist_dir: Directory containing unassigned netlist templates
        generalize_specs_config_dict: Generalized specs configuration dictionary
        init_param_dict: List of parameter dictionaries to evaluate
        sim_config_dict: Simulation configuration dictionary
        corner_list: List of corners to evaluate

    Returns:
        None (Creates an Excel file with evaluation results)
    """
    # Initialize evaluation dictionary
    evaluation_dict = OrderedDict()

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
    logging.info(f"Initialing!!!Zero sim result: {zero_sim_result}")

    # Iterate through parameter sets
    for param_idx, updated_param in enumerate(init_param_dict, 1):
        param_key = f"param_set_{param_idx}"
        evaluation_dict[param_key] = {}

        # Iterate through corners
        for corner_tag in corner_list:
            # Create working directory
            working_dir = create_work_dir(base_folder)

            try:
                # Run simulation for current parameter set and corner
                sim_result = single_evaluation(
                    working_dir=working_dir,
                    unassigned_netlist_dir=unassigned_netlist_dir,
                    updated_param=updated_param,
                    sim_config_dict=sim_config_dict,
                    zero_sim_result=zero_sim_result,
                    corner_tag=corner_tag
                )

                # Store results
                evaluation_dict[param_key][corner_tag] = {
                    "param": updated_param,
                    "result": sim_result
                }

            except Exception as e:
                logging.error(f"Simulation failed for {param_key}, corner {corner_tag}: {str(e)}")
                evaluation_dict[param_key][corner_tag] = {
                    "param": updated_param,
                    "result": zero_sim_result
                }

    # Convert evaluation_dict to DataFrame
    _create_excel_report(evaluation_dict, base_folder)


def _create_excel_report(evaluation_dict: Dict[str, Any], base_folder: str) -> None:
    """
    Create Excel report from evaluation dictionary.

    Args:
        evaluation_dict: Dictionary containing evaluation results
        base_folder: Base directory to save the Excel file

    Returns:
        None
    """
    rows = []

    # Process each parameter set and corner
    for param_set, corner_data in evaluation_dict.items():
        for corner, data in corner_data.items():
            row = {
                'param_set': param_set,
                'corner': corner
            }

            # Add parameters
            for param_key, param_value in data['param'].items():
                row[f"param_{param_key}"] = param_value

            # Add results (skip first level of sim results)
            for sim_type, sim_metrics in data['result'].items():
                for metric_name, metric_value in sim_metrics.items():
                    row[f"result_{metric_name}"] = metric_value

            rows.append(row)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(rows)

    # Reorder columns to ensure param_set is first
    cols = df.columns.tolist()
    cols.remove('param_set')
    cols = ['param_set'] + cols

    df = df[cols]

    # Save to Excel
    output_file = os.path.join(base_folder, 'evaluation_results.xlsx')
    df.to_excel(output_file, index=False)
    logging.info(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    base_folder = '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/run_test/'
    unassigned_netlist_dir = '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_AXS_Corner/'

    generalize_specs_config_path = '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/config/config_AXS_Simple/generalize_specs.yaml'
    with open(generalize_specs_config_path, 'r') as file:
        generalize_specs_config_dict = yaml.safe_load(file)

    init_param_dict_path = '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/config/config_AXS_Simple/init_batch.yaml'
    with open(init_param_dict_path, 'r') as file:
        init_param_dict = yaml.safe_load(file)

    sim_config_dict_path = '/home/wuhan/AnalogDesignAuto/AnalogDesignAuto_MultiAgent/custom_env/config/config_AXS_Simple/simulation.yaml'
    with open(sim_config_dict_path, 'r') as file:
        sim_config_dict = yaml.safe_load(file)

    corner_list = ['tt']
    corner_pattern_1 = ['fs','sf','ff','ss']
    cornet_pattern_2 = ['ss','ff']
    temp_pattern = ['b40','125']
    # Generate a corner summary list: {corner_pattern_1}_{cornet_pattern_2}_{cornet_pattern_2}_{temp_pattern}
    corner_summary = [f"{corner1}_{corner2}_{corner3}_{temp}" for corner1 in corner_pattern_1 for corner2 in cornet_pattern_2 for corner3 in cornet_pattern_2 for temp in temp_pattern]
    corner_list.extend(corner_summary)
    print(f"Evaluation Corner List: {corner_list}")

    batch_evaluation(base_folder, unassigned_netlist_dir, generalize_specs_config_dict, init_param_dict, sim_config_dict, corner_list)