import copy
import logging
from util_func import create_work_dir, retry_decorator
from run_spectre_simulation import run_dynamic_simulation
from assign_param2netlist import update_netlist



def step_simulation_process(region_extract_tag, run_root_dir, valid_param, step_num, dc_check_tag, sim_config_dict,
                            updated_param, unassigned_netlist_dir, zero_sim_result, sim_output_enable_tag,
                            dynamic_queue_tag, norm_specs, norm_ideal_specs, agents, ideal_specs, cal_reward,
                            operation_region_dict=None):
    sim_result = None

    if region_extract_tag:
        if operation_region_dict is None:
            logging.error("Error: region_extract is True but operation_region_dict is None.")
            raise ValueError("operation_region_dict is None when region_extract is True")

    # Create working directory
    working_dir_step = create_work_dir(run_root_dir)
    if valid_param:
        logging.info(f"Step!!!Working directory: {working_dir_step} with step number: {step_num} "
                     f"with region_extract & dc_check enabled and valid param.")
    else:
        logging.info(f"Step!!!Working directory: {working_dir_step} with step number: {step_num} "
                     f"with region_extract: {region_extract_tag} and dc_check: {dc_check_tag}")

    update_netlist(working_dir_step, sim_config_dict, updated_param, unassigned_netlist_dir)

    # Run spectre simulation and normalize the result
    # Define a private function for retrying
    @retry_decorator(retry_count=2, delay_seconds=0.5, default_value=zero_sim_result)
    def _run_simulation_with_retry():
        return run_dynamic_simulation(working_dir_step, sim_config_dict, zero_sim_result,
                                      sim_output_enable_tag, dynamic_queue_tag)

    try:
        sim_result = copy.deepcopy(_run_simulation_with_retry())
    except Exception as e:
        logging.warning(f"Step Warning!!!: {e}. sim_result is {sim_result}."
                        f" Simulation failed, use zero result instead.")
        sim_result = copy.deepcopy(zero_sim_result)

    logging.info(f"Step!!!Simulation result: {sim_result} with step number: {step_num}")
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

    return working_dir_step, observations, sim_result, rew_single