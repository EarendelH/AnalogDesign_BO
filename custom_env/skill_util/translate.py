import yaml
import re
import subprocess
import os
import pty
import select
import sys
import time


def determine_param_mapping(instance_name, config_folder):
    print(f"Debug, config_folder is {config_folder}")
    # List scs file in the file directory
    scs_files = [f for f in os.listdir(config_folder) if f.endswith('.scs')]
    print(f"Debug, scs_files is {scs_files}")
    if not scs_files:
        print(f"Error: No .scs files found in {scs_files} directory.")
        sys.exit(1)

    scs_files = scs_files[0]  # Only consider the first .scs file
    scs_file_path = os.path.join(config_folder, scs_files)

    with open(scs_file_path, 'r') as file:
        content = file.read()

    # Extract the parameter mapping
    instance_match = re.search(r'^\s*(' + re.escape(instance_name) + r')\s\(', content, re.MULTILINE)
    if instance_match:
        line_start = instance_match.start()
        line_end = content.find('\n', line_start)
        line = content[line_start:line_end]

        if f'multi=nf_{instance_name}' in line:
            return {"w": "w", "l": "l", "nf": "simM"}
        elif f'nf=nf_{instance_name}' in line:
            return {"w": "w", "l": "l", "nf": "fingers"}
        else:
            print(f"Error: Unable to determine parameter mapping for instance {instance_name}.")
            sys.exit(1)
    else:
        print(f"Error: Instance {instance_name} not found in the .scs file.")
        sys.exit(1)


def parse_listLibraryCellviews_output(output):
    """
    Parse the output of listLibraryCellviews and return core_cell_list and tb_cell_list.

    Args:
    output (str): Output string from listLibraryCellviews command

    Returns:
    tuple: (core_cell_list, tb_cell_list)
    """
    core_cell_list = []
    tb_cell_list = []
    current_cell = None

    for line in output.split('\n'):
        if line.strip().startswith("Cell:"):
            current_cell = line.split(":")[1].strip()
        elif line.strip().startswith("View: schematic"):
            if current_cell:
                if current_cell.startswith("tb_"):
                    tb_cell_list.append(current_cell)
                else:
                    core_cell_list.append(current_cell)

    return core_cell_list, tb_cell_list


def get_instances_for_cellview(master, lib_name, cell_name):
    """
    Get all instances in a specified cellview.

    Args:
    master: Master file descriptor for Virtuoso session
    lib_name (str): Library name
    cell_name (str): Cell name

    Returns:
    list: List of instance names
    """
    command = f'showCellViewInstances("{lib_name}" "{cell_name}" "schematic")'
    output = send_skill_command(master, command)
    return extract_instances_from_skill_output(output)


def match_instances_to_cellviews(yaml_instances, cellview_instances):
    """
    Match instances from YAML to cellviews.

    Args:
    yaml_instances (list): List of instances extracted from YAML
    cellview_instances (dict): Dictionary of cellviews and their instances

    Returns:
    dict: Mapping of instance names to cellviews
    """
    instance_to_cellview = {}
    for instance in yaml_instances:
        matching_cellviews = []
        for cellview, instances in cellview_instances.items():
            if instance in instances:
                matching_cellviews.append(cellview)

        if len(matching_cellviews) > 1:
            print(f"Error: Instance {instance} found in multiple cellviews: {matching_cellviews}")
            sys.exit(1)
        elif len(matching_cellviews) == 1:
            instance_to_cellview[instance] = matching_cellviews[0]

    return instance_to_cellview


def extract_instances_from_yaml(data):
    """
    Extract instance names from the YAML file.

    Args:
    data (dict): Parsed YAML data

    Returns:
    list: List of instance names found in the YAML file
    """

    core_data = data.get('Core_Param', {})
    instances = set()

    for key in core_data:
        match = re.match(r'(w|l|nf)_(M\d+|MP)(?:_per_finger)?', key)
        if match:
            instances.add(match.group(2))
        else:
            instances.add(key)

    return list(instances)


def extract_instances_from_skill_output(output):
    """
    Extract instance names from Skill command output.

    Args:
    output (str): Output string from Skill command

    Returns:
    list: List of instance names found in the Skill output
    """
    instances = []
    for line in output.split('\n'):
        if line.strip().startswith("Instance:"):
            instances.append(line.split(":")[1].strip())
    return instances


def generate_skill_commands(yaml_data, instance_to_cellview, tb_cell_list, config_folder):
    """
    Generate Skill commands based on YAML file content, instance to cellview mapping, and testbench cell list.

    Args:
    yaml_data (dict): Parsed YAML data
    instance_to_cellview (dict): Mapping of instance names to cellviews
    tb_cell_list (list): List of testbench cells

    Returns:
    list: List of Skill commands to modify instance parameters
    """

    core_data = yaml_data.get('Core_Param', {})
    testbench_data = yaml_data.get('Testbench_Param', {})
    lib_name = yaml_data.get('Lib', '')

    skill_commands = []

    def add_command(lib, cell, view, instance, param, value):
        cell = instance_to_cellview.get(instance, cell)
        command = f'ModifyInstanceParameter("{lib}" "{cell}" "{view}" "{instance}" "{param}" "{value}")'
        skill_commands.append(command)
        print(f"Debug: Adding command: {command}")  # Debug print

    # Generate commands for Core_Param
    for key, value in core_data.items():
        if key.startswith(('C', 'R', 'L')):
            add_command(lib_name, instance_to_cellview.get(key, ''), "schematic", key, key[0].lower(), value)
            continue

        match = re.match(r'(w|l|nf)_(M\w+)(?:_per_finger)?', key)
        if match:
            param_type, instance = match.groups()
            instance = instance.replace('_per_finger', '')
            param_mapping = determine_param_mapping(instance, config_folder)
            skill_param = param_mapping[param_type]
            add_command(lib_name, instance_to_cellview.get(instance, ''), "schematic", instance, skill_param, value)

    # Generate commands for Testbench_Param
    if testbench_data or tb_cell_list:
        for tb_cell in tb_cell_list:
            for instance, value in testbench_data.items():
                if instance.startswith(('V', 'I')):
                    skill_param = "vdc" if instance.startswith('V') else "idc"
                    add_command(lib_name, tb_cell, "schematic", instance, skill_param, value)
                if instance.startswith(('C', 'R', 'L')):
                    add_command(lib_name, tb_cell, "schematic", instance, instance[0].lower(), value)
    else:
        print("Debug: Testbench_Param is empty or not present. Skipping testbench parameter modifications.")

    return skill_commands


def start_virtuoso_session():
    """
    Start a Virtuoso session using a pseudo-terminal.

    Returns:
    tuple: (subprocess.Popen, int) - Virtuoso process and master file descriptor
    """
    try:
        master, slave = pty.openpty()
        process = subprocess.Popen(['virtuoso', '-nograph'],
                                   stdin=slave,
                                   stdout=slave,
                                   stderr=slave,
                                   text=True)
        return process, master
    except Exception as e:
        print(f"Error starting Virtuoso session: {e}")
        return None, None


def wait_for_virtuoso_ready(master, timeout=60):
    """
    Wait for Virtuoso to be ready to accept commands.

    Args:
    master (int): Master file descriptor of the pseudo-terminal
    timeout (int): Maximum time to wait in seconds

    Returns:
    bool: True if Virtuoso is ready, False otherwise
    """
    if master is None:
        print("Error: Invalid master file descriptor")
        return False

    start_time = time.time()
    while True:
        try:
            rlist, _, _ = select.select([master], [], [], 0.1)
            if rlist:
                chunk = os.read(master, 1024).decode()
                print(chunk, end='', flush=True)
                if chunk.strip().endswith(">"):
                    print("Virtuoso is ready.")
                    return True
            else:
                if time.time() - start_time > timeout:
                    print(f"Timeout: Virtuoso did not become ready within {timeout} seconds.")
                    return False
        except Exception as e:
            print(f"Error while waiting for Virtuoso: {e}")
            return False


def send_skill_command(master, command):
    """
    Send a Skill command to Virtuoso and return the output.

    Args:
    master (int): Master file descriptor of the pseudo-terminal
    command (str): Skill command to send

    Returns:
    str: Output of the Skill command
    """
    if master is None:
        print("Error: Invalid master file descriptor")
        return ""

    try:
        # Send the command
        os.write(master, (command + '\n').encode())

        # Read the output
        output = ""
        prompt_ready = False
        while not prompt_ready:
            rlist, _, _ = select.select([master], [], [], 0.1)
            if rlist:
                chunk = os.read(master, 1024).decode()
                if chunk:
                    output += chunk
                    print(chunk, end='', flush=True)  # Print output in real-time
                    if chunk.strip().endswith(">"):
                        prompt_ready = True
            else:
                if output.strip().endswith(">"):
                    prompt_ready = True

        # Process the output
        lines = output.strip().split('\n')
        processed_output = []
        command_seen = False
        for line in lines:
            if line.strip() == command:
                if command_seen:
                    continue  # Skip duplicate command echo
                command_seen = True
            if line.strip() != ">":  # Don't include the prompt in the processed output
                processed_output.append(line)

        return '\n'.join(processed_output)
    except Exception as e:
        print(f"Error sending command: {e}")
        return ""


def load_skill_functions(master):
    """
    Load necessary Skill functions into the Virtuoso session.

    Args:
    master (int): Master file descriptor of the pseudo-terminal

    Returns:
    bool: True if all functions were loaded successfully, False otherwise
    """
    output = None
    skill_files = [
        "modifyInstanceParameterWithCallback.il",
        "showCellViewInstances.il",
        "PrintInstanceDetails.il",
        "listLibraryCellviews.il",
        "copyEntireLibrary.il"
    ]
    for file in skill_files:
        load_command = f'load("{file}")'
        output = send_skill_command(master, load_command)
        print(f"Loading {file}: {output}")
    return "Error" not in output


def parse_value(value):
    """
    Parse a string value into a float, handling unit prefixes.

    Args:
    value (str or float): Value to parse

    Returns:
    float: Parsed value
    """
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r"(\d+(?:\.\d+)?)([fpnumkMG])?", str(value))
    if match:
        num, unit = match.groups()
        num = float(num)
        if unit == 'f':
            return num * 1e-15
        elif unit == 'p':
            return num * 1e-12
        elif unit == 'n':
            return num * 1e-9
        elif unit == 'u':
            return num * 1e-6
        elif unit == 'm':
            return num * 1e-3
        elif unit in ['k', 'K']:
            return num * 1e3
        elif unit == 'M':
            return num * 1e6
        elif unit == 'G':
            return num * 1e9
        else:
            return num
    return float(value)


def compare_values(yaml_value, skill_value):
    """
    Compare two values, handling potential unit differences.

    Args:
    yaml_value (str or float): Value from YAML file
    skill_value (str or float): Value from Skill output

    Returns:
    bool: True if values are equal within a small threshold, False otherwise
    """
    yaml_parsed = parse_value(yaml_value)
    skill_parsed = parse_value(skill_value)
    return abs(yaml_parsed - skill_parsed) < 1e-15  # Use a small threshold to handle floating-point errors


def check_instance_parameters(yaml_data, skill_output, instance_name, config_folder):
    """
    Check if instance parameters in Skill match those in YAML.

    Args:
    yaml_data (dict): Parsed YAML data
    skill_output (str): Output from Skill PrintInstanceDetails command
    instance_name (str): Name of the instance being checked

    Returns:
    bool: True if all parameters match, False otherwise
    """
    yaml_params = yaml_data['Core_Param']
    skill_params = {}
    for line in skill_output.split('\n'):
        match = re.match(r'^\s*(\w+):\s*"([^"]*)"', line)
        if match:
            key, value = match.groups()
            skill_params[key] = value

    if instance_name.startswith('M'):
        param_mapping = determine_param_mapping(instance_name, config_folder)
        print(f"Debug, for {instance_name}, the param_mapping is {param_mapping}")

        if param_mapping['nf'] == 'simM':
            params_to_check = {'l': 'l', 'w': 'w', 'nf': 'simM'}
        else:
            params_to_check = {'l': 'l', 'w': 'w', 'nf': 'fingers'}
            # Check wf
            w = parse_value(skill_params['w'])
            fingers = int(skill_params['fingers'])
            calculated_wf = w * fingers
            actual_wf = parse_value(skill_params['wf'])
            if abs(calculated_wf - actual_wf) > 1e-15:
                print(f"Error: wf value mismatch for {instance_name}. Calculated: {calculated_wf}, Actual: {actual_wf}")
                return False
            else:
                print(f"wf value correct for {instance_name}")

        for yaml_key, skill_key in params_to_check.items():
            if yaml_key == "w":
                yaml_value = yaml_params[f'{yaml_key}_{instance_name}_per_finger']
            else:
                yaml_value = yaml_params[f'{yaml_key}_{instance_name}']
            skill_value = skill_params[skill_key]

            if not compare_values(yaml_value, skill_value):
                print(f"Error: {yaml_key} value mismatch for {instance_name}. YAML: {yaml_value}, Skill: {skill_value}")
                return False
            else:
                print(f"{yaml_key} value correct for {instance_name}")

    elif instance_name.startswith(('C', 'R', 'L')):
        param_key = instance_name[0].lower()
        yaml_value = str(yaml_params.get(instance_name, ''))
        skill_value = str(skill_params.get(param_key, ''))

        if not compare_values(yaml_value, skill_value):
            print(f"Error: Value mismatch for {instance_name}. YAML: {yaml_value}, Skill: {skill_value}")
            return False
        else:
            print(f"Value correct for {instance_name}")

    return True


def translate(source_lib, yaml_data, config_folder, master):

    yaml_instances = extract_instances_from_yaml(yaml_data)

    # virtuoso_process, master = start_virtuoso_session()
    #
    # if virtuoso_process is None or master is None:
    #     print("Failed to start Virtuoso session. Exiting.")
    #     return
    #
    # try:
    #     if not wait_for_virtuoso_ready(master):
    #         print("Failed to detect Virtuoso ready state. Exiting.")
    #         return
    #
    #     print("Virtuoso is ready to accept commands.")
    #
    #     if not load_skill_functions(master):
    #         print("Failed to load Skill functions. Exiting.")
    #         return

    try:
        # Copy new lib
        target_lib = f"{source_lib}_{yaml_data['index']}_{yaml_data['id']}"
        copy_command = f'copyEntireLibrary("{source_lib}" "{target_lib}")'
        output = send_skill_command(master, copy_command)
        print(f"Library copy output: {output}")

        # Run listLibraryCellviews
        list_command = f'listLibraryCellviews("{target_lib}")'
        output = send_skill_command(master, list_command)
        core_cell_list, tb_cell_list = parse_listLibraryCellviews_output(output)
        print("Core cells:", core_cell_list)
        print("Testbench cells:", tb_cell_list)

        # Get instances for each core cell
        cellview_instances = {}
        for cell in core_cell_list:
            instances = get_instances_for_cellview(master, target_lib, cell)
            cellview_instances[cell] = instances

        # Match YAML instances to cellviews
        instance_to_cellview = match_instances_to_cellviews(yaml_instances, cellview_instances)
        print(f"Map dict is {instance_to_cellview}")

        # Generate and send Skill commands
        commands = generate_skill_commands(yaml_data, instance_to_cellview, tb_cell_list, config_folder)
        for command in commands:
            print(f"\nSending command: {command}")
            send_skill_command(master, command)
            print()  # Add a blank line for better readability

        # Check parameters for each instance
        for instance in yaml_instances:
            cellview = instance_to_cellview.get(instance)
            if cellview:
                print(f"Checking parameters for instance {instance} in {cellview}...")
                command = f'PrintInstanceDetails("{target_lib}" "{cellview}" "schematic" "{instance}")'
                output = send_skill_command(master, command)
                if not check_instance_parameters(yaml_data, output, instance):
                    print(f"Parameter mismatch for instance {instance}.")
            else:
                print(f"Warning: No matching cellview found for instance {instance}.")

        print("All instance parameters checked. Proceeding with parameter modifications.")

        send_skill_command(master, "exit")
    except Exception as e:
        print(f"An error occurred: {e}")
