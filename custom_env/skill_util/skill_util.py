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


def start_virtuoso_session(interactive=False):
    if interactive:
        master, slave = pty.openpty()
        process = subprocess.Popen(['virtuoso', '-nograph'],
                                   stdin=slave,
                                   stdout=slave,
                                   stderr=slave,
                                   text=True)
        return process, master
    else:
        process = subprocess.Popen(['virtuoso', '-nograph'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        return process, None


def send_skill_command(process, command):
    # Send a Skill command to Virtuoso
    process.stdin.write(command + '\n')
    process.stdin.flush()

    # Wait for the command to complete (you might need to adjust the waiting mechanism)
    time.sleep(0.5)

    # Read the output (optional, depending on your needs)
    output = process.stdout.readline()
    return output


def load_skill_functions(process, master=None, interactive=False):
    file_path = os.path.join(os.path.dirname(__file__), "modifyInstanceParameterWithCallback.il")
    load_command = f'load("{file_path}")'
    output = send_skill_command(process, load_command, master, interactive)
    if not interactive:
        print(f"Loading Skill functions: {output.strip()}")


def main(interactive=False):

    parser = argparse.ArgumentParser(description='Interface with Cadence Virtuoso')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode')
    args = parser.parse_args()

    yaml_path = os.path.join(os.path.dirname(__file__), "init_param_demo.yaml")
    commands = generate_skill_commands(yaml_path)

    virtuoso_process, master = start_virtuoso_session(args.interactive)

    try:
        load_skill_functions(virtuoso_process, master, args.interactive)

        for command in commands:
            print(f"Sending command: {command}")
            output = send_skill_command(virtuoso_process, command, master, args.interactive)
            if not args.interactive:
                print(f"Output: {output.strip()}")

        send_skill_command(virtuoso_process, "exit", master, args.interactive)
    finally:
        virtuoso_process.terminate()
        if args.interactive:
            os.close(master)


if __name__ == "__main__":
    main()