import copy
import os
import subprocess
from importlib import import_module
import pickle
import logging
from util.extract_device_param_value import extract_operation_region_w_name


def run_spectre_simulation(work_dir, sim_config, show_output=False):
    """
    Run spectre simulation based on the given config and netlist
    :param work_dir: working directory
    :param sim_config: config simulation item and corresponding result parse function
    :param show_output: show the output of the simulation
    :return: Arranged simulation results
    """
    results = {}

    for simulation_config in sim_config:
        simulation = simulation_config["simulation_name"]
        assigned_netlist_filename = f"{simulation}.scs"

        # Check if the assigned netlist file exists
        file_list = os.listdir(work_dir)
        if assigned_netlist_filename not in file_list:
            raise ValueError(f"Assigned netlist file {assigned_netlist_filename} not found.")

        # Run spectre simulation
        # print(f"Execute command: spectre -64 {os.path.join(work_dir, assigned_netlist_filename)}")
        # print(f"Run spectre simulation for: {simulation}")
        if show_output:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}", shell=True)
        else:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Process the simulation files as specified in the config
        raw_dir = os.path.join(work_dir, f"{simulation}.raw")
        sim_result_file = simulation_config["simulation_file"]
        if not isinstance(sim_result_file, list):
            sim_result_file = [sim_result_file]
        parse_funcs = simulation_config["parse_func"]
        if not isinstance(parse_funcs, list):
            parse_funcs = [parse_funcs]

        # Convert binary file to text file
        for idx, sim_file in enumerate(sim_result_file):
            file_to_process = os.path.join(raw_dir, sim_file)
            processed_file = f"{file_to_process}.encode"
            subprocess.run(f"psf {file_to_process} -o {processed_file}", shell=True)
            # print(f"Processed file: {sim_file}")

            # Load the function to process the results and execute it
            module_name = f"find{simulation}"
            function_name = parse_funcs[idx]
            module = import_module(f"util.{module_name}")
            # print(f"Debug!!!Current Path: {current_path}")
            # print(f"Debug!!!Work Dict: {work_dir}")
            # print(f"Debug!!!Processing file: {processed_file}")
            # Apply absolute path for avoiding file not found error
            processed_file_full_path = os.path.join(raw_dir, processed_file)
            function = getattr(module, function_name)
            result = function(processed_file_full_path)
            results[simulation] = result

    return results

# Test Code
# work_dir = "../netlist_assign_test"
# sim_config_path = "../config_3/simulation.yaml"
# with open(sim_config_path, 'r') as file:
#     sim_config = yaml.safe_load(file)
# test_result = run_spectre_simulation(work_dir, sim_config)
# print(test_result)

# Output {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': nan, 'gainBandWidth': nan}, 'Trans': {
# 'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081}, 'PSRR': {'powerSupplyRejectionRatio':
# 9.24983891912481}}


def run_dynamic_simulation(work_dir, sim_config, zero_sim_result, show_output=False, dynamic_queue=True):
    """
    Run spectre simulation. Once the output result is zero (for pwr, reset value is 1), the simulation will be stopped.
    Zero simulation result is given.
    :param work_dir: working directory
    :param sim_config: config simulation item and corresponding result parse function
    :param show_output: show the output of the simulation
    :param zero_sim_result: zero simulation result
    :param dynamic_queue: if True, use dynamic queue mechanism; if False, run all simulations
    :return: Arranged simulation results
    """

    results = copy.deepcopy(zero_sim_result)
    fail_tag = False

    for simulation_config in sim_config:
        simulation = simulation_config["simulation_name"]
        assigned_netlist_name = simulation_config[f"netlist_name"]
        assigned_netlist_filename = f"{assigned_netlist_name}.scs"
        objective = simulation_config["objective"]
        logging.debug(f"Debug!!! Running Simulation: {simulation}")

        # Check if the assigned netlist file exists
        file_list = os.listdir(work_dir)
        logging.debug(f"Debug!!! File List: {file_list}")
        if assigned_netlist_filename not in file_list:
            raise ValueError(f"Assigned netlist file {assigned_netlist_filename} not found.")

        # Run spectre simulation
        logging.debug(f"Execute command: spectre -64 {os.path.join(work_dir, assigned_netlist_filename)}")
        logging.debug(f"Run spectre simulation for: {simulation}")
        if show_output:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}", shell=True)
        else:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Process the simulation files as specified in the config
        raw_dir = os.path.join(work_dir, f"{assigned_netlist_name}.raw")
        sim_result_file = simulation_config["simulation_file"]
        # Convert to list if it is not
        if not isinstance(sim_result_file, list):
            sim_result_file = [sim_result_file]
        parse_funcs = simulation_config["parse_func"]
        if not isinstance(parse_funcs, list):
            parse_funcs = [parse_funcs]

        # Convert binary file to text file
        for idx, sim_file in enumerate(sim_result_file):
            file_to_process = os.path.join(raw_dir, sim_file)
            processed_file = f"{file_to_process}.encode"
            subprocess.run(f"psf {file_to_process} -o {processed_file}", shell=True)
            logging.debug(f"Debug!!! Processed file: {sim_file}")

            # Load the function to process the results and execute it
            script_name = simulation_config["script_name"]
            function_name = parse_funcs[idx]
            module = import_module(f"util.{script_name}")
            logging.debug(f"Debug!!!Work Dict: {work_dir}")
            logging.debug(f"Debug!!!Processing file: {processed_file}")
            # Apply absolute path for avoiding file not found error
            processed_file_full_path = os.path.join(raw_dir, processed_file)
            function = getattr(module, function_name)
            result = function(processed_file_full_path)
            results[simulation] = result

            logging.debug(f"Debug in run_dynamic_simulation.py, result in {simulation} is {result}")

            # Add Simulation Name before each keys in the result dictionary. Avoid error in flatten the dictionary
            modified_result = {f"{simulation}_{key}": value for key, value in result.items()}
            results[simulation] = modified_result

            if dynamic_queue:
                if any(value == 100.0 for value in result.values()) and (objective == "min"):
                    fail_tag = True
                    logging.warning(f"Simulation {simulation} failed. Return 100.0")
                    logging.warning(f"Partial result success: {results}")
                if any(value == 0.0 for value in result.values()) and (objective == "max"):
                    fail_tag = True
                    logging.warning(f"Simulation {simulation} failed. Return 0.0")
                    logging.warning(f"Partial result success: {results}")

        # Break the loop if fail_tag is True
        if fail_tag and dynamic_queue:
            logging.warning(f"Debug!!! Simulation {simulation} failed. Break the loop. No more simulation should be run.")
            break

    logging.debug(f"Debug!!! results: {results} and fail_tag: {fail_tag}")

    return results


def run_dynamic_simulation_singularity(work_dir, sim_config, zero_sim_result, show_output=False, dynamic_queue=True):
    """
    在超算环境中使用Singularity容器运行Spectre仿真。
    :param work_dir: 工作目录
    :param sim_config: 配置仿真项目和相应的结果解析函数
    :param zero_sim_result: 零仿真结果
    :param container_path: Singularity容器路径，例如"~/container/centos_spectre.sif"
    :param show_output: 是否显示仿真输出
    :param dynamic_queue: 如果为True，使用动态队列机制；如果为False，运行所有仿真
    :return: 整理后的仿真结果
    """
    results = copy.deepcopy(zero_sim_result)
    fail_tag = False

    container_path = "/home/lizz_lab/cse12310401/container/centos_spectre.sif"

    for simulation_config in sim_config:
        simulation = simulation_config["simulation_name"]
        assigned_netlist_name = simulation_config[f"netlist_name"]
        assigned_netlist_filename = f"{assigned_netlist_name}.scs"
        objective = simulation_config["objective"]
        logging.debug(f"Debug!!! Running Simulation: {simulation}")

        # 检查指定的网表文件是否存在
        file_list = os.listdir(work_dir)
        logging.debug(f"Debug!!! File List: {file_list}")
        if assigned_netlist_filename not in file_list:
            raise ValueError(f"Assigned netlist file {assigned_netlist_filename} not found.")

        # 使用Singularity运行Spectre仿真
        scs_file_path = os.path.join(work_dir, assigned_netlist_filename)
        singularity_cmd = f"singularity exec --bind $HOME {container_path} spectre -64 {scs_file_path}"
        logging.debug(f"Execute command: {singularity_cmd}")
        logging.debug(f"Run spectre simulation for: {simulation}")
        
        if show_output:
            subprocess.run(singularity_cmd, shell=True)
        else:
            subprocess.run(singularity_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 处理仿真文件
        raw_dir = os.path.join(work_dir, f"{assigned_netlist_name}.raw")
        sim_result_file = simulation_config["simulation_file"]
        # 如果不是列表则转换为列表
        if not isinstance(sim_result_file, list):
            sim_result_file = [sim_result_file]
        parse_funcs = simulation_config["parse_func"]
        if not isinstance(parse_funcs, list):
            parse_funcs = [parse_funcs]

        # 将二进制文件转换为文本文件
        for idx, sim_file in enumerate(sim_result_file):
            file_to_process = os.path.join(raw_dir, sim_file)
            processed_file = f"{file_to_process}.encode"
            
            # 使用Singularity运行psf命令
            psf_cmd = f"singularity exec --bind $HOME {container_path} psf {file_to_process} -o {processed_file}"
            subprocess.run(psf_cmd, shell=True)
            logging.debug(f"Debug!!! Processed file: {sim_file}")

            # 加载处理结果的函数并执行
            script_name = simulation_config["script_name"]
            function_name = parse_funcs[idx]
            module = import_module(f"util.{script_name}")
            logging.debug(f"Debug!!!Work Dict: {work_dir}")
            logging.debug(f"Debug!!!Processing file: {processed_file}")
            # 应用绝对路径以避免文件未找到错误
            processed_file_full_path = os.path.join(raw_dir, processed_file)
            function = getattr(module, function_name)
            result = function(processed_file_full_path)
            results[simulation] = result

            logging.debug(f"Debug in run_dynamic_simulation_singularity, result in {simulation} is {result}")

            # 在结果字典中的每个键前添加仿真名称，避免展平字典时出错
            modified_result = {f"{simulation}_{key}": value for key, value in result.items()}
            results[simulation] = modified_result

            if dynamic_queue:
                if any(value == 100.0 for value in result.values()) and (objective == "min"):
                    fail_tag = True
                    logging.warning(f"Simulation {simulation} failed. Return 100.0")
                    logging.warning(f"Partial result success: {results}")
                if any(value == 0.0 for value in result.values()) and (objective == "max"):
                    fail_tag = True
                    logging.warning(f"Simulation {simulation} failed. Return 0.0")
                    logging.warning(f"Partial result success: {results}")

        # 如果fail_tag为True则中断循环
        if fail_tag and dynamic_queue:
            logging.warning(f"Debug!!! Simulation {simulation} failed. Break the loop. No more simulation should be run.")
            break

    logging.debug(f"Debug!!! results: {results} and fail_tag: {fail_tag}")

    return results


def run_dynamic_simulation_psfascii(work_dir, sim_config, zero_sim_result, show_output=False, dynamic_queue=True):
    """
    Run spectre simulation. Once the output result is zero (for pwr, reset value is 1), the simulation will be stopped.
    Zero simulation result is given.
    :param work_dir: working directory
    :param sim_config: config simulation item and corresponding result parse function
    :param show_output: show the output of the simulation
    :param zero_sim_result: zero simulation result
    :param dynamic_queue: if True, use dynamic queue mechanism; if False, run all simulations
    :return: Arranged simulation results
    """

    results = copy.deepcopy(zero_sim_result)
    fail_tag = False

    for simulation_config in sim_config:
        simulation = simulation_config["simulation_name"]
        assigned_netlist_name = simulation_config[f"netlist_name"]
        assigned_netlist_filename = f"{assigned_netlist_name}.scs"
        objective = simulation_config["objective"]
        logging.debug(f"Debug!!! Running Simulation: {simulation}")

        # Check if the assigned netlist file exists
        file_list = os.listdir(work_dir)
        logging.debug(f"Debug!!! File List: {file_list}")
        if assigned_netlist_filename not in file_list:
            raise ValueError(f"Assigned netlist file {assigned_netlist_filename} not found.")

        # Run spectre simulation
        logging.debug(f"Execute command: spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)} +escchars -format psfascii +aps=conservative +spice +logstatus")
        logging.debug(f"Run spectre simulation for: {simulation}")
        if show_output:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)} +escchars -format psfascii +aps=conservative +spice +logstatus", shell=True)
        else:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)} +escchars -format psfascii +aps=conservative +spice +logstatus",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Process the simulation files as specified in the config
        raw_dir = os.path.join(work_dir, f"{assigned_netlist_name}.raw")
        sim_result_file = simulation_config["simulation_file"]
        # Convert to list if it is not
        if not isinstance(sim_result_file, list):
            sim_result_file = [sim_result_file]
        parse_funcs = simulation_config["parse_func"]
        if not isinstance(parse_funcs, list):
            parse_funcs = [parse_funcs]

        # Convert binary file to text file
        for idx, sim_file in enumerate(sim_result_file):
            file_to_process = os.path.join(raw_dir, sim_file)
            # processed_file = f"{file_to_process}.encode"
            # subprocess.run(f"psf {file_to_process} -o {processed_file}", shell=True)
            logging.debug(f"Debug!!! Processed file: {sim_file}")

            # Load the function to process the results and execute it
            script_name = simulation_config["script_name"]
            function_name = parse_funcs[idx]
            module = import_module(f"util.{script_name}")
            logging.debug(f"Debug!!!Work Dict: {work_dir}")
            logging.debug(f"Debug!!!Processing file: {file_to_process}")
            # Apply absolute path for avoiding file not found error
            processed_file_full_path = os.path.join(raw_dir, file_to_process)
            function = getattr(module, function_name)
            result = function(processed_file_full_path)
            results[simulation] = result

            logging.debug(f"Debug in run_dynamic_simulation.py, result in {simulation} is {result}")

            # Add Simulation Name before each keys in the result dictionary. Avoid error in flatten the dictionary
            modified_result = {f"{simulation}_{key}": value for key, value in result.items()}
            results[simulation] = modified_result

            if dynamic_queue:
                if any(value == 100.0 for value in result.values()) and (objective == "min"):
                    fail_tag = True
                    logging.warning(f"Simulation {simulation} failed. Return 100.0")
                    logging.warning(f"Partial result success: {results}")
                if any(value == 0.0 for value in result.values()) and (objective == "max"):
                    fail_tag = True
                    logging.warning(f"Simulation {simulation} failed. Return 0.0")
                    logging.warning(f"Partial result success: {results}")

        # Break the loop if fail_tag is True
        if fail_tag and dynamic_queue:
            logging.warning(f"Debug!!! Simulation {simulation} failed. Break the loop. No more simulation should be run.")
            break

    logging.debug(f"Debug!!! results: {results} and fail_tag: {fail_tag}")

    return results


def run_region_simulation(work_dir, sim_config, show_output=False):
    """
    Run spectre simulation. Once the output result is zero (for pwr, reset value is 1), the simulation will be stopped.
    Zero simulation result is given.
    :param work_dir: working directory
    :param sim_config: config simulation item and corresponding result parse function
    :param show_output: show the output of the simulation
    :return: Arranged simulation results
    """

    reset_operation_region_dict = {}

    for simulation_config in sim_config:
        assigned_netlist_name = simulation_config[f"netlist_name"]
        assigned_netlist_filename = f"{assigned_netlist_name}.scs"

        # Check if the assigned netlist file exists
        file_list = os.listdir(work_dir)
        # print(f"Debug!!! File List: {file_list}")
        if assigned_netlist_filename not in file_list:
            raise ValueError(f"Assigned netlist file {assigned_netlist_filename} not found.")

        if show_output:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}", shell=True)
        else:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        dc_reset_raw_result_path = os.path.join(work_dir, "Region.raw/dcOpInfo.info")
        dc_reset_result_path = os.path.join(work_dir, "Region.raw/dcOpInfo.info.encode")
        subprocess.run(f"psf {dc_reset_raw_result_path} -o {dc_reset_result_path}", shell=True)
        reset_operation_region_dict = extract_operation_region_w_name(dc_reset_result_path)

    return reset_operation_region_dict


def run_region_simulation_psfascii(work_dir, sim_config, show_output=False):
    """
    Run spectre simulation. Once the output result is zero (for pwr, reset value is 1), the simulation will be stopped.
    Zero simulation result is given.
    :param work_dir: working directory
    :param sim_config: config simulation item and corresponding result parse function
    :param show_output: show the output of the simulation
    :return: Arranged simulation results
    """

    reset_operation_region_dict = {}

    for simulation_config in sim_config:
        assigned_netlist_name = simulation_config[f"netlist_name"]
        assigned_netlist_filename = f"{assigned_netlist_name}.scs"

        # Check if the assigned netlist file exists
        file_list = os.listdir(work_dir)
        # print(f"Debug!!! File List: {file_list}")
        if assigned_netlist_filename not in file_list:
            raise ValueError(f"Assigned netlist file {assigned_netlist_filename} not found.")

        if show_output:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}", shell=True)
        else:
            subprocess.run(f"spectre -64 ++aps {os.path.join(work_dir, assigned_netlist_filename)}",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        dc_reset_raw_result_path = os.path.join(work_dir, "Region.raw/dcOpInfo.info")
        # dc_reset_result_path = os.path.join(work_dir, "Region.raw/dcOpInfo.info.encode")
        # subprocess.run(f"psf {dc_reset_raw_result_path} -o {dc_reset_result_path}", shell=True)
        reset_operation_region_dict = extract_operation_region_w_name(dc_reset_raw_result_path)

    return reset_operation_region_dict