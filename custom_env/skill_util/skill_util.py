import yaml
import re
import subprocess
import time
import os
import pty
import select
import sys
import argparse


def generate_skill_commands(yaml_file):
    # Read the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    core_data = data.get('Core_Param', {})
    skill_commands = []
    lib_name = data.get('Lib', {})
    core_cell_name = data.get('Core_Cell', {})

    for key, value in core_data.items():
        # Extract instance name and parameter
        match = re.match(r'(w|l|nf)_(M\d+)(?:_per_finger)?', key)
        if match:
            param_type, instance = match.groups()

            # Generate Skill command based on parameter type
            if param_type == 'w':
                skill_param = "w"
            elif param_type == 'l':
                skill_param = "l"
            elif param_type == 'nf':
                if instance == "MP":
                    skill_param = "simM"
                else:
                    skill_param = "fingers"
            elif param_type == 'c' and instance.startswith('C'):
                skill_param = "c"
            elif param_type == 'r' and instance.startswith('R'):
                skill_param = "r"
            else:
                continue  # Skip if it doesn't match any known patterns

            # Create Skill command
            command = (f'ModifyInstanceParameter("{lib_name}" "{core_cell_name}" '
                       f'"schematic" "{instance}" "{skill_param}" "{value}")')
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
        os.write(master, (command + '\n').encode())
        output = ""
        while True:
            rlist, _, _ = select.select([master], [], [], 0.1)
            if rlist:
                chunk = os.read(master, 1024).decode()
                print(chunk, end='', flush=True)
                output += chunk
                if output.endswith("> "):
                    break
        return output
    except Exception as e:
        print(f"Error sending command: {e}")
        return ""


def load_skill_functions(master):

    # if running folder do not exist modifyInstanceParameterWithCallback.il
    # Cp the file from script folder to running folder

    load_command = f'load("modifyInstanceParameterWithCallback.il")'
    output = send_skill_command(master, load_command)
    print(f"Loading Skill functions: {output}")
    return "Error" not in output


def main():

    yaml_path = os.path.join(os.path.dirname(__file__), "init_param_demo.yaml")
    commands = generate_skill_commands(yaml_path)
    virtuoso_process, master = start_virtuoso_session()

    if virtuoso_process is None or master is None:
        print("Failed to start Virtuoso session. Exiting.")
        return

    try:
        # Wait for Virtuoso to start and be ready
        initial_output = wait_for_virtuoso_ready(master)
        print("Virtuoso is ready to accept commands.")

        # Load necessary Skill functions
        if not load_skill_functions(master):
            print("Failed to load Skill functions. Exiting.")
            return

        # Send each command to Virtuoso
        for command in commands:
            print(f"Sending command: {command}")
            output = send_skill_command(master, command)
            print(f"Output: {output}")

        # Close the Virtuoso session
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
