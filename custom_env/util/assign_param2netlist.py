from collections import OrderedDict
import yaml
import re
import os


def assign_param2netlist(param_dict, unassigned_netlist_file_path, assigned_netlist_file_path):
    """
    Assign parameter values to netlist file.
    :param param_dict: Dictionary of parameter values.
    :param unassigned_netlist_file_path: Path to unassigned netlist file.
    :param assigned_netlist_file_path: Path to assigned netlist file.
        """

    with open(unassigned_netlist_file_path, 'r') as f:
        unassigned_netlist_line = f.readlines()

    # Extract unset variable names from SCS file
    parameters_line = ''
    for line in unassigned_netlist_line:
        if line.strip().startswith('parameters'):
            parameters_line = line.strip()
            break
    param_str = parameters_line.split('parameters')[1].strip()
    param_names_in_netlist = re.split(r'\s+', param_str)
    unset_params = [param for param in param_names_in_netlist if "=" not in param]

    # Extract variable names from param_dict and compare
    param_dict_list = list(param_dict.keys())
    # Delete "_per_finger" in param_dict_list
    # param_dict_list = [key.replace('_per_finger', '') if key.endswith('_per_finger') else key for key in
    #                    param_dict_list]
    if set(unset_params) != set(param_dict_list):
        raise ValueError(
            f"Mismatch between SCS and Assigned variables. Missing in YAML: {set(unset_params) - set(param_dict_list)}, Missing in SCS: {set(param_dict_list) - set(unset_params)}")

    # Validate "nf_" value in param_dict, str is original data type. Check the value is int in format.
    for key, value in param_dict.items():
        if "nf_" in key:
            try:
                int(value)
            except ValueError:
                raise ValueError(f"{key} in YAML should be an integer, got {value}.")

    # Update param_dict using param_dict values, delete "w_X_per_finger" in param_dict_list and generate "w_X" value.
    # key_to_delete = []
    # new_dict_entries = []
    # param_dict_processed = param_dict.copy()
    # for key, value in param_dict_processed.items():
    #     if key.endswith('_per_finger'):
    #         index = key.split('_')[1]
    #         nf_key = f"nf_{index}"
    #         nf_value = int(param_dict_processed[nf_key])
    #
    #         value_numeric, unit = float(value[:-1]), value[-1]
    #
    #         new_value = value_numeric * nf_value
    #
    #         new_key = f"w_{index}"
    #         new_dict_entries.append((new_key, f"{new_value}{unit}"))
    #         key_to_delete.append(key)
    #
    # for key, value in new_dict_entries:
    #     param_dict_processed[key] = value
    #
    # for key in key_to_delete:
    #     del param_dict_processed[key]

    # print(param_dict_processed)

    update_params = []
    for param in param_names_in_netlist:
        if param in param_dict:
            update_params.append(f"{param}={param_dict[param]}")
        else:
            update_params.append(param)
    updated_parameters_line = f"parameters {' '.join(update_params)}"

    # Validate all variable are assigned.
    assigned_param_name = re.split(r'\s+', updated_parameters_line.split('parameters')[1].strip())
    unassigned_params_after_update = [param for param in assigned_param_name if "=" not in param]
    # print(unassigned_params_after_update)
    if unassigned_params_after_update:
        raise ValueError(f"Unassigned parameters: {unassigned_params_after_update}")

    # Save assigned netlist file.
    for i, line in enumerate(unassigned_netlist_line):
        if line.strip().startswith("parameters"):
            unassigned_netlist_line[i] = updated_parameters_line + "\n"
            break
    with open(assigned_netlist_file_path, 'w') as f:
        f.writelines(unassigned_netlist_line)

    return assigned_netlist_file_path


# Test Code

# param_dict = OrderedDict([
#     ('multi_C0', '36.0'), ('multi_C2', '4.0'), ('w_M115_per_finger', '5.8u'), ('l_M115', '3.0u'), ('nf_M115', '3'), ('w_M114_per_finger', '5.8u'), ('l_M114', '3.0u'), ('nf_M114', '3'), ('w_M113_per_finger', '5.8u'), ('l_M113', '3.0u'), ('nf_M113', '3'), ('w_M112_per_finger', '5.8u'), ('l_M112', '3.0u'), ('nf_M112', '3'), ('w_M111_per_finger', '5.8u'), ('l_M111', '3.0u'), ('nf_M111', '3'), ('w_M110_per_finger', '5.8u'), ('l_M110', '3.0u'), ('nf_M110', '3'), ('w_M103_per_finger', '5.8u'), ('l_M103', '3.0u'), ('nf_M103', '3'), ('w_M104_per_finger', '5.8u'), ('l_M104', '3.0u'), ('nf_M104', '3'), ('w_M105_per_finger', '5.8u'), ('l_M105', '3.0u'), ('nf_M105', '3'), ('w_M106_per_finger', '5.8u'), ('l_M106', '3.0u'), ('nf_M106', '3'), ('w_M107_per_finger', '5.8u'), ('l_M107', '3.0u'), ('nf_M107', '3'), ('w_M108_per_finger', '5.8u'), ('l_M108', '3.0u'), ('nf_M108', '3'), ('w_M109_per_finger', '5.8u'), ('l_M109', '3.0u'), ('nf_M109', '3'), ('w_M116_per_finger', '5.8u'), ('l_M116', '3.0u'), ('nf_M116', '3'), ('w_M101_per_finger', '5.8u'), ('l_M101', '3.0u'), ('nf_M101', '3'), ('w_M102_per_finger', '5.8u'), ('l_M102', '3.0u'), ('nf_M102', '3'), ('w_M201_per_finger', '2.0u'), ('l_M201', '3.0u'), ('nf_M201', '3'), ('w_M202_per_finger', '2.0u'), ('l_M202', '3.0u'), ('nf_M202', '3'), ('w_M203_per_finger', '2.0u'), ('l_M203', '3.0u'), ('nf_M203', '3'), ('w_M204_per_finger', '2.0u'), ('l_M204', '3.0u'), ('nf_M204', '3'), ('w_M205_per_finger', '2.0u'), ('l_M205', '3.0u'), ('nf_M205', '3'), ('w_M206_per_finger', '2.0u'), ('l_M206', '3.0u'), ('nf_M206', '3'), ('w_M50_per_finger', '2.0u'), ('l_M50', '0.8u'), ('nf_M50', '2'), ('w_M49_per_finger', '5.4u'), ('l_M49', '0.8u'), ('nf_M49', '13'), ('w_M36_per_finger', '6.0u'), ('l_M36', '3.0u'), ('nf_M36', '3'), ('w_M35_per_finger', '2.9u'), ('l_M35', '1.2u'), ('nf_M35', '1'), ('w_M23_per_finger', '2.2u'), ('l_M23', '2.6u'), ('nf_M23', '2'), ('w_M19_per_finger', '6.0u'), ('l_M19', '1.1u'), ('nf_M19', '1'), ('w_M47_per_finger', '4.5u'), ('l_M47', '0.9u'), ('nf_M47', '3'), ('w_M43_per_finger', '5.8u'), ('l_M43', '1.1u'), ('nf_M43', '3'), ('w_M42_per_finger', '2.0u'), ('l_M42', '0.8u'), ('nf_M42', '2'), ('w_M39_per_finger', '1.1u'), ('l_M39', '1.0u'), ('nf_M39', '2'), ('w_M25_per_finger', '6.0u'), ('l_M25', '3.0u'), ('nf_M25', '5'), ('w_M18_per_finger', '6.0u'), ('l_M18', '1.1u'), ('nf_M18', '1'), ('w_M20_per_finger', '6.0u'), ('l_M20', '1.1u'), ('nf_M20', '3'), ('w_M26_per_finger', '6.0u'), ('l_M26', '3.0u'), ('nf_M26', '5'), ('w_M24_per_finger', '6.0u'), ('l_M24', '3.0u'), ('nf_M24', '2'), ('w_M22_per_finger', '1.9u'), ('l_M22', '1.5u'), ('nf_M22', '2'), ('w_M21_per_finger', '3.0u'), ('l_M21', '3.4u'), ('nf_M21', '2'), ('w_M28_per_finger', '2.9u'), ('l_M28', '1.2u'), ('nf_M28', '15'), ('w_M51_per_finger', '4.7u'), ('l_M51', '1.3u'), ('nf_M51', '3'), ('w_M46_per_finger', '2.5u'), ('l_M46', '0.7u'), ('nf_M46', '2'), ('w_M44_per_finger', '3.0u'), ('l_M44', '0.7u'), ('nf_M44', '3'), ('w_M41_per_finger', '5.7u'), ('l_M41', '0.7u'), ('nf_M41', '3'), ('w_M40_per_finger', '5.7u'), ('l_M40', '1.1u'), ('nf_M40', '3'), ('w_M34_per_finger', '1.7u'), ('l_M34', '1.1u'), ('nf_M34', '6'), ('w_M10_per_finger', '2.0u'), ('l_M10', '2.8u'), ('nf_M10', '3'), ('w_M3_per_finger', '8.0u'), ('l_M3', '1.9u'), ('nf_M3', '3'), ('w_M2_per_finger', '8.0u'), ('l_M2', '1.9u'), ('nf_M2', '1'), ('w_M11_per_finger', '5.4u'), ('l_M11', '3.0u'), ('nf_M11', '3'), ('w_M5_per_finger', '4.7u'), ('l_M5', '3.0u'), ('nf_M5', '3'), ('w_M9_per_finger', '6.0u'), ('l_M9', '1.0u'), ('nf_M9', '3'), ('w_M1_per_finger', '8.0u'), ('l_M1', '1.9u'), ('nf_M1', '3'), ('w_M16_per_finger', '2.5u'), ('l_M16', '0.8u'), ('nf_M16', '8'), ('w_M15_per_finger', '2.5u'), ('l_M15', '0.8u'), ('nf_M15', '8'), ('w_M7_per_finger', '38.0u'), ('l_M7', '0.7u'), ('nf_M7', '120'), ('w_M6_per_finger', '1.7u'), ('l_M6', '0.8u'), ('nf_M6', '1'), ('w_M0_per_finger', '8.0u'), ('l_M0', '1.9u'), ('nf_M0', '1'), ('w_M4_per_finger', '4.7u'), ('l_M4', '3.0u'), ('nf_M4', '3')
# ])
# unassigned_netlist_file_path = ("/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_AXS/Load_Reg_Trans_parameterized.scs")
# assigned_netlist_file_path = ("/Users/hanwu/Downloads/AXS_Trans/Netlist/Trans_10.scs")
# assign_param2netlist(param_dict, unassigned_netlist_file_path, assigned_netlist_file_path)

def update_netlist(work_dir, sim_config, param_dict, unassigned_netlist_dir_path, tag=''):
    """
    Update the netlist file based on the given parameter dictionary
    :param work_dir: working directory
    :param sim_config: simulation configuration
    :param param_dict: parameter dictionary
    :param unassigned_netlist_dir_path: unassigned netlist file path
    :param tag: additional tag (string), default is empty string
    :return: Assigned SCS file
    """
    for simulation_config in sim_config:

        sim_name = simulation_config["simulation_name"]

        if tag:
            unassigned_netlist_filename = f"{sim_name}_parameterized_{tag}.scs"
        else:
            unassigned_netlist_filename = f"{sim_name}_parameterized.scs"

        assigned_netlist_filename = f"{sim_name}.scs"
        unassigned_netlist_file_path = os.path.join(unassigned_netlist_dir_path, unassigned_netlist_filename)
        assigned_netlist_file_path = os.path.join(work_dir, assigned_netlist_filename)
        assign_param2netlist(param_dict, unassigned_netlist_file_path, assigned_netlist_file_path)

        if not os.path.exists(assigned_netlist_file_path):
            print(f"Netlist file assign fail, try to assign again.")
            assign_param2netlist(param_dict, unassigned_netlist_file_path, assigned_netlist_file_path)
            if not os.path.exists(assigned_netlist_file_path):
                raise ValueError(f"Netlist file assign fail, please check.")


# Test Code
# work_dir = "/Users/hanwu/Downloads/LDO_SSF"
# sim_config_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/simulation.yaml"
# with open(sim_config_path, 'r') as file:
#     sim_config = yaml.safe_load(file)
# param_dict_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/init_param.yaml"
# with open(param_dict_path, 'r') as file:
#     param_dict = yaml.safe_load(file)
# unassigned_netlist_dir_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template"
# update_netlist(work_dir, sim_config, param_dict, unassigned_netlist_dir_path)
