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
#     ('C1', '3.7p'), ('C2', '7.3p'), ('C3', '7.4p'), ('IB', '1.5u'), ('R1', '700.0k'), ('w_M16_per_finger', '10.0u'), ('l_M16', '3.5u'), ('nf_M16', '4'), ('w_M21_per_finger', '8.75u'), ('l_M21', '1.0u'), ('nf_M21', '8'), ('w_M24_per_finger', '10.0u'), ('l_M24', '3.5u'), ('nf_M24', '10'), ('w_M12_per_finger', '4.5u'), ('l_M12', '1.25u'), ('nf_M12', '3'), ('w_M11_per_finger', '4.5u'), ('l_M11', '1.25u'), ('nf_M11', '7'), ('w_M5_per_finger', '7.5u'), ('l_M5', '6.5u'), ('nf_M5', '3'), ('w_M4_per_finger', '2.75u'), ('l_M4', '1.25u'), ('nf_M4', '1'), ('w_M3_per_finger', '2.75u'), ('l_M3', '1.25u'), ('nf_M3', '1'), ('w_M33_per_finger', '10.0u'), ('l_M33', '3.5u'), ('nf_M33', '7'), ('w_M0_per_finger', '17.5u'), ('l_M0', '60.0n'), ('nf_M0', '6'), ('w_M31_per_finger', '2.5u'), ('l_M31', '9.75u'), ('nf_M31', '8'), ('w_M32_per_finger', '2.5u'), ('l_M32', '9.75u'), ('nf_M32', '6'), ('w_M15_per_finger', '4.0u'), ('l_M15', '9.0u'), ('nf_M15', '3'), ('w_M22_per_finger', '2.75u'), ('l_M22', '0.5u'), ('nf_M22', '8'), ('w_M23_per_finger', '2.75u'), ('l_M23', '0.5u'), ('nf_M23', '2'), ('w_M14_per_finger', '2.5u'), ('l_M14', '9.75u'), ('nf_M14', '2'), ('w_M13_per_finger', '2.5u'), ('l_M13', '9.75u'), ('nf_M13', '7'), ('w_M6_per_finger', '2.5u'), ('l_M6', '9.75u'), ('nf_M6', '5'), ('w_M2_per_finger', '2.75u'), ('l_M2', '8.25u'), ('nf_M2', '3'), ('w_M1_per_finger', '2.75u'), ('l_M1', '8.25u'), ('nf_M1', '3')
# ])
# unassigned_netlist_file_path = ("/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_Jianping_N65/Trans_1_2V_parameterized.scs")
# assigned_netlist_file_path = ("/Users/hanwu/Downloads/Trans/Jianping_Trans/Netlist/Trans_10.scs")
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
