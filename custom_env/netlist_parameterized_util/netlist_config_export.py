import yaml


def export_netlist_config(netlist_path, config_value_assign_path, config_range_define_path):
    with open(netlist_path, 'r') as f:
        netlist_content = f.readlines()

    subckt_lines = []
    inside_subckt = False
    for line in netlist_content:
        line = line.strip()
        if "subckt" in line:
            inside_subckt = True
        if inside_subckt:
            subckt_lines.append(line)
        if "ends" in line:
            inside_subckt = False
            break

    instances = []
    for line in subckt_lines[1:]:
        if 'nch_mac' in line or 'pch_mac' in line:
            instances.append(line.strip())

    instance_info = []
    for instance in instances:
        tokens = instance.split()
        instance_name = tokens[0]
        instance_type = [t for t in tokens if "nch_mac" in t or "pch_mac" in t][0] if any(
            "nch_mac" in t or "pch_mac" in t for t in tokens) else "Unknown"
        instance_info.append((instance_name, instance_type))

    yaml_dict = {}
    param_templates = {
        'nch_mac': ['w_{0}', 'l_{0}', 'nf_{0}'],
        'pch_mac': ['w_{0}', 'l_{0}', 'nf_{0}']
    }
    for instance_name, instance_type in instance_info:
        variable_names = []
        if instance_type in param_templates.keys():
            for template in param_templates[instance_type]:
                variable_name = template.format(instance_name)
                variable_names.append({"variable_name": variable_name, "value": None})
        yaml_dict[instance_name] = {
            "instance_type": instance_type,
            "params": variable_names
        }

    parameters_line = None
    for line in netlist_content:
        if line.strip().startswith("parameters"):
            parameters_line = line.strip()
            break

    parameter_tokens = parameters_line.split()[1:]
    undefined_parameters = []
    for token in parameter_tokens:
        if "=" not in token:
            variable_name = token
            if all(variable_name != p['variable_name'] for instance in yaml_dict.values() for p in instance['params']):
                undefined_parameters.append(variable_name)

    yaml_dict["other_variable"] = {
        "instance_type": "other_variable",
        "params": [{"variable_name": param, "value": None} for param in undefined_parameters]
    }

    with open(config_value_assign_path, 'w') as f:
        f.write(yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False))

    yaml_dict_with_range_step = {}

    for instance_name, instance_info in yaml_dict.items():
        param_names = []
        for param in instance_info['params']:
            variable_name = param['variable_name']
            param_names.append({"variable_name": variable_name, "value": {"range": None, "step": None}})

        yaml_dict_with_range_step[instance_name] = {
            "instance_type": instance_info['instance_type'],
            "params": param_names
        }

    yaml_str_with_range_step = yaml.dump(yaml_dict_with_range_step, default_flow_style=False, sort_keys=False)
    with open(config_range_define_path, 'w') as f:
        f.write(yaml_str_with_range_step)


export_netlist_config("netlistParameterize/netlist_parameterized/Trans_parameterized.scs",
                      "netlistParameterize/config_value_template.yaml",
                      "netlistParameterize/config_value_range_define_template.yaml")
