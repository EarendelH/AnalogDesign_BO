import os
import argparse
import yaml
import sys
from collect_value import collect_value
from translate import (start_virtuoso_session, wait_for_virtuoso_ready, load_skill_functions, send_skill_command,
                       translate, close_virtuoso_session)


def process_single_yaml(source_lib, yaml_data, input_dir):
    virtuoso_process, master = start_virtuoso_session()

    if virtuoso_process is None or master is None:
        print("Failed to start Virtuoso session. Exiting.")
        sys.exit(1)

    try:
        if not wait_for_virtuoso_ready(master):
            print("Failed to detect Virtuoso ready state. Exiting.")
            sys.exit(1)

        print("Virtuoso is ready to accept commands.")

        if not load_skill_functions(master):
            print("Failed to load Skill functions. Exiting.")
            sys.exit(1)

        # close_command = '(hiDeleteForm "techSaveDrmForm")'
        # send_skill_command(master, close_command)

        translate(source_lib, yaml_data, input_dir, master)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        close_virtuoso_session(virtuoso_process, master)


def main(input_dir):
    collect_value(input_dir)

    info_yaml_path = os.path.join(input_dir, 'info.yaml')
    with open(info_yaml_path, 'r') as f:
        info_yaml = yaml.safe_load(f)

    source_lib = info_yaml['Lib']

    value_dir = os.path.join(input_dir, 'value')
    for yaml_file in os.listdir(value_dir):
        yaml_path = os.path.join(value_dir, yaml_file)
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        print(f"Processing {yaml_file}...")
        process_single_yaml(source_lib, yaml_data, input_dir)
        print(f"Finished processing {yaml_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign values to Virtuoso")
    parser.add_argument("input_dir", help="Input directory containing Excel and YAML files")
    args = parser.parse_args()

    main(args.input_dir)
