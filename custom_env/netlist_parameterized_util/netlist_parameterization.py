import csv
import re
import yaml


def normalize_spacing(instance):
    # Remove extra spaces and tabs
    instance = re.sub(r'\s+', ' ', instance).strip()

    # Make sure only one space exists between attributes
    attributes = instance.split(" ")
    normalized_instance = " ".join(attributes)

    return normalized_instance


def netlist_parameterization(netlist_path, output_netlist_path):
    """
    Modify a netlist file.

    Parameters:
        netlist_path (str): The path to the original netlist file.
        output_netlist_path (str): The path to save the modified netlist file.
        output_csv_path (str): The path to save the CSV file containing instance names and types.

    Returns:
        None
    """
    # Read the original netlist file
    with open(netlist_path, 'r') as f:
        lines = f.readlines()

    # Initialize variables
    parameters_line = ''
    subckt_content_lines = []
    in_subckt = False
    indent_unit = "    "  # Assuming 4 spaces are used for indentation
    config_data = {}

    for line in lines:
        if line.strip().startswith("parameters"):
            parameters_line = line.strip()
        elif line.strip().startswith("subckt"):
            in_subckt = True
        elif line.strip().startswith("ends"):
            in_subckt = False

        if in_subckt:
            subckt_content_lines.append(line)

    # Extract instance names and types
    instance_info = []
    instances = []
    current_instance_lines = []
    pattern = re.compile(r'^\s+[A-Z]')

    for line in subckt_content_lines[1:]:  # Skip the first 'subckt' line
        match = pattern.match(line)
        if match:
            if current_instance_lines:
                instances.append(" ".join(current_instance_lines).replace("\\", "").strip())
            current_instance_lines = [line.strip()]
        else:
            if line.strip():
                current_instance_lines.append(line.strip())

    if current_instance_lines:
        instances.append(" ".join(current_instance_lines).replace("\\", "").strip())

    # Modify w, l, nf attributes and extract instance names and types
    additional_parameters = []
    for i, instance in enumerate(instances):
        instance_name = instance.split()[0]
        instance_type = instance.split(")")[1].split()[0]
        instance_info.append((instance_name, instance_type))

        config_data[instance_name] = {
            'type': instance_type,
            'variables': []
        }
        # Update YAML config file

        if instance_type == "nch_mac" or instance_type == "pch_mac":
            for attr in ["l=", "w=", "nf="]:
                repl = f"{attr[:-1]}_{instance_name}"
                instances[i] = re.sub(f"{attr}[\w\.e\-]+", f"{attr}{repl}", instances[i])
                additional_parameters.append(repl)
                config_data[instance_name]['variables'].append(repl)  # Update YAML config file
    # More instance type can be added here

    # Update 'parameters' line
    parameters_line += " " + " ".join(additional_parameters)

    # Modify ad, as, pd, ps, nrd, nrs, sa, sb attributes

    formulas_summary = {
        "pch_mac": {
            "ad": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((nf_{0}/2)*5.4e-07))*w_{0}",
            "as": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0))*w_{0}",
            "pd": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(((nf_{0}/2)*5.4e-07)*2+nf_{0}*w_{0})",
            "ps": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0)*2+(nf_{0}+2)*w_{0})",
            "nrd": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07/nf_{0}/w_{0})",
            "nrs": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07*2.7e-07*2.7e-07/(2.7e-07*2.7e-07*(nf_{0}-2)+2.7e-07*(2.7e-07+2.7e-07))/w_{0})",
            "sa": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})",
            "sb": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})"
        },
        "nch_mac": {
            "ad": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((nf_{0}/2)*5.4e-07))*w_{0}",
            "as": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0))*w_{0}",
            "pd": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(((nf_{0}/2)*5.4e-07)*2+nf_{0}*w_{0})",
            "ps": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0)*2+(nf_{0}+2)*w_{0})",
            "nrd": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07/nf_{0}/w_{0})",
            "nrs": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07*2.7e-07*2.7e-07/(2.7e-07*2.7e-07*(nf_{0}-2)+2.7e-07*(2.7e-07+2.7e-07))/w_{0})",
            "sa": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})",
            "sb": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})"
        },
    }
    # More instance type can be added here

    for i, instance in enumerate(instances):
        instance_name = instance.split()[0]
        instance_type = instance.split(")")[1].split()[0]
        formulas = formulas_summary.get(instance_type, {})
        for attr, formula in formulas.items():
            formula_instance = formula.format(instance_name)
            instances[i] = re.sub(f"{attr}=[\w\.e\-]+", f"{attr}={formula_instance}", instances[i])

    # Normalize spacing for all instances
    for i in range(len(instances)):
        instances[i] = normalize_spacing(instances[i])

    # Write the modified netlist to a new file
    with open(output_netlist_path, 'w') as f:
        in_subckt = False
        for line in lines:
            if line.strip().startswith("subckt"):
                f.write(line)
                in_subckt = True
            elif line.strip().startswith("ends"):
                in_subckt = False
                # Insert modified instances with proper indentation
                for instance in instances:
                    f.write(indent_unit + instance + "\n")
                f.write(line)
            elif not in_subckt:
                if line.strip().startswith("parameters"):
                    f.write(parameters_line + "\n")
                else:
                    f.write(line)


# Test the function with a sample netlist file
netlist_parameterization('netlistParameterize/netlist_original/Trans.scs',
                         'netlistParameterize/netlist_parameterized/Trans_parameterized.scs')
