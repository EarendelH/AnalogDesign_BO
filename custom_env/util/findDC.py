# from extract_device_param_value import parse_device_values, parse_device_param, find_device_param_value
from util.extract_device_param_value import parse_device_values, parse_device_param, find_device_param_value


def findDCValue(filepath):
    # Extract the properties using the previously defined function

    instance_name = "V0"
    property_name = "pwr"
    default_value = 1.0

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

def extract_dcOP_data(file_path, search_keyword):
    """
    Search the first line in a file containing a specific keyword and extract float data.

    Args:
    - file_path: Path to the file to be processed.
    - search_keyword: Keyword to search for in the file.

    Returns:
    - A float extracted from the line containing the keyword, or None if not found.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if search_keyword in line:
                    # Split the line by space and extract the third element as data
                    parts = line.split()
                    if len(parts) >= 3:  # Ensure there are at least 3 parts
                        extracted_data = float(parts[2])  # Convert to float
                    break  # Stop searching after finding the first match
        if extracted_data is None:
            raise ValueError("Keyword not found or file does not contain valid data format.")
    except FileNotFoundError:
        raise FileNotFoundError("File does not exist.")
    except ValueError as e:
        raise ValueError(f"Error while processing the file: {e}")

    return abs(extracted_data)


def findIQ(filepath):
    search_keyword = "V0:p"
    value = extract_dcOP_data(filepath, search_keyword)
    return {"IQ": value}


# Test Code
# path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOp.dc.encode"
# print(findIQ(path))
