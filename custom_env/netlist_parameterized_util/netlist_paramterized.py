import re
import yaml
import os


def split_transistor_block(lines):
    """Split a block of SPICE lines into sublists, each representing a discrete MOSFET definition.

    Args:
        lines (list): A list of SPICE lines.

    Returns:
        list: A list of sublists, where each sublist represents a single MOSFET definition.
    """

    nested_split_texts = []
    current_part = []
    first_line_indent_nested = None

    # Process each line for nested list structure
    for line in lines:
        if line.strip():  # Skip entirely blank lines
            line_indent = len(line) - len(line.lstrip())

            # Start a new part if the line starts with "M" followed by a digit or first part
            if (line.lstrip().startswith('M') and line.lstrip()[1].isdigit()) or first_line_indent_nested is None:
                # Save current part if it exists
                if current_part:
                    nested_split_texts.append(current_part)
                    current_part = []

                current_part = [line]
                first_line_indent_nested = line_indent
            elif line_indent > first_line_indent_nested:
                # Continue the current part
                current_part.append(line)
            else:
                # Start a new part, save current if exists
                if current_part:
                    nested_split_texts.append(current_part)
                    current_part = [line]
                first_line_indent_nested = line_indent

    # Ensure the last part is added
    if current_part:
        nested_split_texts.append(current_part)

    processed_mosfet_lists = []

    for part in nested_split_texts:
        # Check if the part is a MOSFET definition by looking for lines starting with "M" after stripping spaces
        if part[0].lstrip().startswith('M'):
            # Extract the initial indentation level from the first line
            initial_indent = len(part[0]) - len(part[0].lstrip())
            # Merge the part lines into a single line, separating parameters with a single space
            merged_line = ' '.join([line.strip() for line in part])
            # Reapply the initial indentation to the merged line
            processed_line = ' ' * initial_indent + merged_line
            # Add the processed line to the new list
            processed_mosfet_lists.append([processed_line])
            # Remove \\ in the all line
            processed_mosfet_lists[-1][0] = processed_mosfet_lists[-1][0].replace("\\", "")
            # Separating parameters with a single space
            processed_mosfet_lists[-1][0] = re.sub(r'(?<=[\w\d])\s+(?=[\w\d])', ' ', processed_mosfet_lists[-1][0])
            # Add a newline character to the end of the line
            processed_mosfet_lists[-1][0] += '\n'
        else:
            # For non-MOSFET parts, simply add them to the new list as they are
            processed_mosfet_lists.append(part)

    return processed_mosfet_lists


# Test Code
# file_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.scs"
# with open(file_path, 'r') as f:
#     lines = f.readlines()
# print(split_transistor_block(lines))


def netlist_parameterized(input_scs_path, output_scs_path, output_yaml_path):
    param_line = None

    with open(input_scs_path, 'r') as f:
        lines = f.readlines()

    data_dict = {
        'other_variable':
            {
                'instance_type': 'other_variable',
                'params': []
            }
    }

    is_param_line = False

    for line in lines:
        if line.strip().startswith("parameters"):
            is_param_line = True
            param_line = line
            break

    if is_param_line:
        params = param_line.split()[1:]
        for param in params:
            if "=" not in param:
                sub_dict = {
                    'variable_name': param,
                    'value': {
                        'range': None,
                        'step': None
                    }
                }

                data_dict['other_variable']['params'].append(sub_dict)

    # Modify SCS file
    netlist_parts = split_transistor_block(lines)

    device_info = {}

    formulas_summary = {
        "pch_mac": {
            "w": "w_{0}",
            "l": "l_{0}",
            "nf": "nf_{0}",
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
            "w": "w_{0}",
            "l": "l_{0}",
            "nf": "nf_{0}",
            "ad": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((nf_{0}/2)*5.4e-07))*w_{0}",
            "as": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0))*w_{0}",
            "pd": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(((nf_{0}/2)*5.4e-07)*2+nf_{0}*w_{0})",
            "ps": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0)*2+(nf_{0}+2)*w_{0})",
            "nrd": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07/nf_{0}/w_{0})",
            "nrs": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07*2.7e-07*2.7e-07/(2.7e-07*2.7e-07*(nf_{0}-2)+2.7e-07*(2.7e-07+2.7e-07))/w_{0})",
            "sa": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})",
            "sb": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})"
        },
        "nch_5_mac": {
            "w": "w_{0}",
            "l": "l_{0}",
            "nf": "nf_{0}",
            "ad": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((nf_{0}/2)*5.4e-07))*w_{0}",
            "as": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0))*w_{0}",
            "pd": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(((nf_{0}/2)*5.4e-07)*2+nf_{0}*w_{0})",
            "ps": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0)*2+(nf_{0}+2)*w_{0})",
            "nrd": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07/nf_{0}/w_{0})",
            "nrs": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07*2.7e-07*2.7e-07/(2.7e-07*2.7e-07*(nf_{0}-2)+2.7e-07*(2.7e-07+2.7e-07))/w_{0})",
            "sa": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})",
            "sb": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})"
        },
        "pch_5_mac": {
            "w": "w_{0}",
            "l": "l_{0}",
            "nf": "nf_{0}",
            "ad": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((nf_{0}/2)*5.4e-07))*w_{0}",
            "as": "((nf_{0}-int(nf_{0}/2)*2)*(4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0))*w_{0}",
            "pd": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(((nf_{0}/2)*5.4e-07)*2+nf_{0}*w_{0})",
            "ps": "(nf_{0}-int(nf_{0}/2)*2)*((4.8e-07+((nf_{0}-1)*5.4e-07)/2+0)*2+(nf_{0}+1)*w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*((4.8e-07+4.8e-07+(nf_{0}/2-1)*5.4e-07+0+0)*2+(nf_{0}+2)*w_{0})",
            "nrd": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07/nf_{0}/w_{0})",
            "nrs": "(nf_{0}-int(nf_{0}/2)*2)*(2.7e-07*2.7e-07/(2.7e-07+2.7e-07*(nf_{0}-1))/w_{0})+(nf_{0}+1-int((nf_{0}+1)/2)*2)*(2.7e-07*2.7e-07*2.7e-07/(2.7e-07*2.7e-07*(nf_{0}-2)+2.7e-07*(2.7e-07+2.7e-07))/w_{0})",
            "sa": "1/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})",
            "sb": "/(1 / ((4.8e-07 + l_{0}/2) + (5.4e-07 + l_{0})*(1-1)/3.5 ) )-(0.5*l_{0})"
        },
    }

    for part in netlist_parts:
        if part[0].lstrip().startswith('M'):
            instance_name = part[0].split()[0]
            instance_type = part[0].split(")")[1].split()[0]
            # print(f"Instance Name: {instance_name}, Instance Type: {instance_type}")
            device_info[instance_name] = instance_type
            data_dict[instance_name] = {
                "instance_type": instance_type,
                "params": [{
                    "variable_name": f"w_{instance_name}_per_finger",
                    "value": {
                        "range": None,
                        "step": None
                    }
                },
                    {
                        "variable_name": f"l_{instance_name}",
                        "value": {
                            "range": None,
                            "step": None
                        }
                    },
                    {
                        "variable_name": f"nf_{instance_name}",
                        "value": {
                            "range": None,
                            "step": None
                        }
                    }]
            }
            if instance_type in formulas_summary.keys():
                for param, formula in formulas_summary[instance_type].items():
                    formula_instance = formula.format(instance_name)
                    part[0] = re.sub(f"{param}=[\w\.e\-]+", f"{param}={formula_instance}", part[0])
                # print(f"Modified Part: {part}")

    # Save file
    with open(output_scs_path, 'w') as f:
        for part in netlist_parts:
            # Add l_, w_, nf_ to the parameters line
            if part[0].lstrip().startswith('parameters'):
                part[0] = part[0].strip() + " " + " ".join(
                    [f"w_{instance_name} l_{instance_name} nf_{instance_name}" for instance_name in
                     device_info.keys()]) + "\n"
            f.write(''.join(part))

    # Save data_dict to a YAML file
    with open(output_yaml_path, 'w') as f:
        f.write(yaml.dump(data_dict, default_flow_style=False, sort_keys=False))

    return data_dict


# Test Code
input_scs = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Line_Reg_25m.scs"
output = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/Line_Reg_25m_parameterized.scs"
output_yaml = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/param_range_template.yaml"
dict_out = netlist_parameterized(input_scs, output, output_yaml)
print(dict_out)
