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

    core_data = data.get('Core', {})
    skill_commands = []

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
                skill_param = "fingers"

            # Create Skill command
            command = f'ModifyInstanceParameter("LDO_Cai_Skill" "LDO_Core" "schematic" "{instance}" "{skill_param}" "{value}")'
            skill_commands.append(command)

    return skill_commands


def start_virtuoso_session():
    master, slave = pty.openpty()
    process = subprocess.Popen(['virtuoso', '-nograph'],
                               stdin=slave,
                               stdout=slave,
                               stderr=slave,
                               text=True)
    return process, master


def wait_for_virtuoso_ready(master):
    output = ""
    ready = False
    while not ready:
        rlist, _, _ = select.select([master], [], [], 0.1)
        if rlist:
            chunk = os.read(master, 1024).decode()
            print(chunk, end='', flush=True)
            output += chunk
            if "> t" in output:
                os.write(master, b'\n')
                chunk = os.read(master, 1024).decode()
                print(chunk, end='', flush=True)
                if chunk.strip() == ">":
                    ready = True
    return output


def send_skill_command(master, command):
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


def load_skill_functions(process, master=None, interactive=False):

    load_command = f'load("modifyInstanceParameterWithCallback")'
    output = send_skill_command(master, load_command)
    print(f"Loading Skill functions: {output}")


def main():

    yaml_path = os.path.join(os.path.dirname(__file__), "init_param_demo.yaml")
    commands = generate_skill_commands(yaml_path)
    virtuoso_process, master = start_virtuoso_session()

    try:
        # Wait for Virtuoso to start and be ready
        initial_output = wait_for_virtuoso_ready(master)
        print("Virtuoso is ready to accept commands.")

        # Load necessary Skill functions
        load_skill_functions(master)

        # Send each command to Virtuoso
        for command in commands:
            print(f"Sending command: {command}")
            output = send_skill_command(master, command)
            print(f"Output: {output}")

        # Close the Virtuoso session
        send_skill_command(master, "exit")
    finally:
        virtuoso_process.terminate()
        os.close(master)

if __name__ == "__main__":
    main()