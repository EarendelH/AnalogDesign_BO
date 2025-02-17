import os
import logging
from util.assign_param2netlist import update_netlist
from util.util_func import retry_decorator
from util.run_spectre_simulation import run_dynamic_simulation_psfascii
import copy

def single_evaluation(working_dir, unassigned_netlist_dir, updated_param, sim_config_dict, zero_sim_result, corner_tag):

    sim_result = None
    sim_output_enable_tag = False
    dynamic_queue_tag = False

    os.makedirs(working_dir, exist_ok=True)
    logging.debug(f"Working_dir: {working_dir}")

    update_netlist(working_dir, sim_config_dict, updated_param, unassigned_netlist_dir, corner_tag)

    @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=zero_sim_result)
    def _run_simulation_with_retry():
        return run_dynamic_simulation_psfascii(working_dir, sim_config_dict, zero_sim_result,
                                               sim_output_enable_tag, dynamic_queue_tag)

    try:
        sim_result = copy.deepcopy(_run_simulation_with_retry())
    except Exception as e:
        logging.warning(f"Step Warning!!!: {e}. sim_result is {sim_result}."
                        f" Simulation failed, use zero result instead.")
        sim_result = copy.deepcopy(zero_sim_result)

    logging.info(f"Step!!!Simulation result: {sim_result} in corner: {corner_tag}")

    return updated_param, sim_result