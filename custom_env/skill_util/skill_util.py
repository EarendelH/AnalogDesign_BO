import yaml
import re
import subprocess
import os
import pty
import select


def extract_instances_from_yaml(yaml_file):
    # Read the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    core_data = data.get('Core_Param', {})
    instances = set()

    for key in core_data.keys():
        match = re.match(r'(w|l|nf)_(M\d+|MP)(?:_per_finger)?', key)
        if match:
            # For w, l, nf parameters, extract the instance name
            instances.add(match.group(2))
        else:
            # For other parameters, use the key itself as the instance name
            instances.add(key)

    return list(instances)


def extract_instances_from_skill_output(output):
    instances = []
    for line in output.split('\n'):
        if line.strip().startswith("Instance:"):
            instances.append(line.split(":")[1].strip())
    return instances


def generate_skill_commands(yaml_file):
    # Read the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    core_data = data.get('Core_Param', {})
    skill_commands = []
    lib_name = data.get('Lib', {})
    core_cell_name = data.get('Core_Cell', {})

    for key, value in core_data.items():
        # Handle capacitors and resistors directly
        if key.startswith('C') or key.startswith('R'):
            instance = key
            skill_param = 'c' if key.startswith('C') else 'r'
            command = f'ModifyInstanceParameter("{lib_name}" "{core_cell_name}" "schematic" "{instance}" "{skill_param}" "{value}")'
            skill_commands.append(command)
            continue

        # Handle other components (transistors)
        match = re.match(r'(w|l|nf)_(M\d+|MP)(?:_per_finger)?', key)
        if match:
            param_type, instance = match.groups()

            # Generate Skill command based on parameter type and instance name
            if param_type == 'w':
                skill_param = "w"
            elif param_type == 'l':
                skill_param = "l"
            elif param_type == 'nf':
                if instance == "MP":
                    skill_param = "simM"
                else:
                    skill_param = "fingers"

            # Create Skill command
            command = f'ModifyInstanceParameter("{lib_name}" "{core_cell_name}" "schematic" "{instance}" "{skill_param}" "{value}")'
            skill_commands.append(command)

    return skill_commands


def start_virtuoso_session():
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


def wait_for_virtuoso_ready(master):
    if master is None:
        print("Error: Invalid master file descriptor")
        return ""

    output = ""
    ready = False
    while not ready:
        try:
            rlist, _, _ = select.select([master], [], [], 0.1)
            if rlist:
                chunk = os.read(master, 1024).decode()
                print(chunk, end='', flush=True)
                output += chunk
                if "> t" in output:
                    os.write(master, b'\n')  # Send an enter
                    chunk = os.read(master, 1024).decode()
                    print(chunk, end='', flush=True)
                    if chunk.strip() == ">":
                        ready = True
        except Exception as e:
            print(f"Error while waiting for Virtuoso: {e}")
            return output
    return output


def send_skill_command(master, command):
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
    # Load modifyInstanceParameterWithCallback.il
    load_command = 'load("modifyInstanceParameterWithCallback.il")'
    output = send_skill_command(master, load_command)
    print(f"Loading modifyInstanceParameterWithCallback.il: {output}")

    # Load showCellViewInstances.il
    load_command = 'load("showCellViewInstances.il")'
    output = send_skill_command(master, load_command)
    print(f"Loading showCellViewInstances.il: {output}")

    # Load PrintInstanceDetails.il
    load_command = 'load("PrintInstanceDetails.il")'
    output = send_skill_command(master, load_command)
    print(f"Loading PrintInstanceDetails.il: {output}")

    return "Error" not in output


def parse_value(value):
    match = re.match(r"(\d+(?:\.\d+)?)(p|n|u|m)?", value)
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
        elif unit == 'k':
            return num * 1e3
        elif unit == 'M':
            return num * 1e6
        elif unit == 'G':
            return num * 1e9
        else:
            return num
    return float(value)


def compare_values(yaml_value, skill_value):
    yaml_parsed = parse_value(yaml_value)
    skill_parsed = parse_value(skill_value)
    return abs(yaml_parsed - skill_parsed) < 1e-15  # Use a small threshold to handle floating-point errors


def check_instance_parameters(yaml_data, skill_output, instance_name):
    yaml_params = yaml_data['Core_Param']
    skill_params = {}

    for line in skill_output.split('\n'):
        match = re.match(r'^\s*(\w+):\s*"([^"]*)"', line)
        if match:
            key, value = match.groups()
            skill_params[key] = value

    if instance_name.startswith('M'):
        if instance_name == 'MP':
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

        for yaml_key, skill_key in params_to_check.items():
            if yaml_key == "w":
                yaml_value = yaml_params[f'{yaml_key}_{instance_name}_per_finger']
            else:
                yaml_value = yaml_params[f'{yaml_key}_{instance_name}']
            skill_value = skill_params[skill_key]
            if not compare_values(yaml_value, skill_value):
                print(f"Error: {yaml_key} value mismatch for {instance_name}. YAML: {yaml_value}, Skill: {skill_value}")
                return False

    elif instance_name.startswith(('C', 'R')):
        if instance_name.startswith('C'):
            param_key = 'c'
        elif instance_name.startswith('R'):
            param_key = 'r'
        yaml_value = yaml_params[instance_name]
        skill_value = skill_params[param_key]
        if not compare_values(yaml_value, skill_value):
            print(f"Error: Value mismatch for {instance_name}. YAML: {yaml_value}, Skill: {skill_value}")
            return False

    return True


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "init_param_demo.yaml")

    # Extract instances from YAML
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    yaml_instances = extract_instances_from_yaml(yaml_path)
    print("Instances extracted from YAML:", yaml_instances)

    virtuoso_process, master = start_virtuoso_session()

    if virtuoso_process is None or master is None:
        print("Failed to start Virtuoso session. Exiting.")
        return

    try:
        initial_output = wait_for_virtuoso_ready(master)
        print("Virtuoso is ready to accept commands.")

        if not load_skill_functions(master):
            print("Failed to load Skill functions. Exiting.")
            return

        lib_name = yaml_data.get('Lib', '')
        core_cell_name = yaml_data.get('Core_Cell', '')

        if not lib_name or not core_cell_name:
            print("Error: Lib or Core_Cell not found in YAML file. Exiting.")
            return

        # Run showCellViewInstances
        show_command = f'showCellViewInstances("{lib_name}" "{core_cell_name}" "schematic")'
        output = send_skill_command(master, show_command)

        # Extract instances from Skill output
        skill_instances = extract_instances_from_skill_output(output)
        print("Instances extracted from Skill output:", skill_instances)

        # Compare instances
        missing_instances = set(yaml_instances) - set(skill_instances)
        if missing_instances:
            print("Error: The following instances are missing in the schematic:")
            for instance in missing_instances:
                print(f"  - {instance}")
            print("Exiting program.")
            return

        print("All instances from YAML are present in the schematic. Proceeding with parameter checks.")

        # Check parameters for each instance
        for instance in yaml_instances:
            print(f"Checking parameters for instance {instance}...")
            command = f'PrintInstanceDetails("{lib_name}" "{core_cell_name}" "schematic" "{instance}")'
            output = send_skill_command(master, command)
            if not check_instance_parameters(yaml_data, output, instance):
                print(f"Parameter mismatch for instance {instance}. Exiting program.")
                return

        print("All instance parameters match. Proceeding with parameter modifications.")

        # Continue with the rest of the script (parameter modifications)
        commands = generate_skill_commands(yaml_path)
        for command in commands:
            print(f"\nSending command: {command}")
            output = send_skill_command(master, command)
            print()  # Add a blank line for better readability

        send_skill_command(master, "exit")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if virtuoso_process:
            virtuoso_process.terminate()
        if master:
            os.close(master)


if __name__ == "__main__":
    main()