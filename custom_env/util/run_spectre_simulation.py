import os
import subprocess


def run_spectre_simulation(work_dir, sim_config):
    """
    Run spectre simulation based on the given config and netlist
    :param work_dir: working directory
    :param sim_config: config simulation item and corresponding result parse function
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
        print(f"Execute command: spectre -64 {os.path.join(work_dir, assigned_netlist_filename)}")
        print(f"Run spectre simulation for: {simulation}")
        subprocess.run(f"spectre -64 {os.path.join(work_dir, assigned_netlist_filename)}", shell=True)

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
            print(f"Processed file: {sim_file}")

            # Load the function to process the results and execute it
            module_name = f"find{simulation}"
            function_name = parse_funcs[idx]
            module = __import__(module_name)
            function = getattr(module, function_name)
            result = function(processed_file)
            results[simulation] = result

    return results
