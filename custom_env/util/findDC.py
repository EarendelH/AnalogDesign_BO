# from extract_device_param_value import parse_device_values, parse_device_param, find_device_param_value
from util.extract_device_param_value import parse_device_values, parse_device_param, find_device_param_value


def findDCValue(filepath):
    # Extract the properties using the previously defined function

    instance_name = "V0"
    property_name = "pwr"
    default_value = 1

    try:
        value_dict = parse_device_values(filepath)
        param_dict = parse_device_param(filepath)
        value = find_device_param_value(instance_name, property_name, value_dict, param_dict)
        value = abs(value)

        # Check if the value is a valid result
        if isinstance(value, str):
            raise ValueError("Error in finding device parameter value")

    except Exception as e:
        print(f"Warning: {e}. Setting value to default ({default_value}).")
        value = default_value

    return {property_name: value}


# Test Code
# file_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# print(findDCValue(file_path))
