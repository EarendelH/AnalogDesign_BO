from util.util_device_info import parse_device_info
import re


def parse_instance_properties(filepath, device_info):
    # Read the file content
    with open(filepath, "r") as file:
        file_content = file.read()

    # Extract the VALUE section
    match = re.search(r'VALUE(.+)', file_content, re.DOTALL)
    if not match:
        raise ValueError("The 'VALUE' section not found in the file.")

    value_section = match.group(1)

    # Split the section into individual device values based on the provided delimiter
    device_values_raw = re.split(r'\"[\w.]+?\" \"\w+?\" \(', value_section)
    device_values_raw = [v for v in device_values_raw if v.strip()]

    # Create a dictionary to store the parsed values
    parsed_values = {}

    # Iterate over the parsed device info to extract corresponding values
    for device in device_info:
        device_name_pattern = r'\"([\w.]+?)\" \"' + device['device_name'] + r'\" \('
        device_name_matches = re.findall(device_name_pattern, value_section)

        # If there are no instances of a particular device, continue to the next device
        if not device_name_matches:
            continue

        # Extract values for the properties of each device instance
        for instance_name in device_name_matches:
            instance_values = {}
            instance_pattern = r'\"' + instance_name + r'\" \"' + device['device_name'] + r'\" \(([\s\S]+?)\n\)'
            instance_match = re.search(instance_pattern, value_section)

            if instance_match:
                instance_value_str = instance_match.group(1)
                instance_value_list = [v.strip() for v in instance_value_str.split('\n') if v.strip()]

                for i, prop in enumerate(device['properties']):
                    if i < len(instance_value_list):
                        prop_value = instance_value_list[i]
                        # Convert the value based on its type
                        if prop['property_type'] == 'FLOAT DOUBLE':
                            try:
                                prop_value = float(prop_value)
                            except ValueError:
                                prop_value = None  # For values like "nan"
                        elif prop['property_type'] == 'INT BYTE':
                            prop_value = int(prop_value)

                        instance_values[prop['property_name']] = prop_value

            parsed_values[instance_name] = instance_values

    return parsed_values


def extract_instance_properties(filepath):
    # Using the previously defined parseDeviceInfo function
    device_info = parse_device_info(filepath)

    # Read the file content
    with open(filepath, "r") as file:
        file_content = file.read()

    # Extracting the device instance names and their types from the VALUE section
    device_instance_types = re.findall(r'\"([\w.]+?)\" \"(\w+?)\" \(', file_content)

    # Creating a dictionary for instance names and their types
    instance_type_dict = {instance: dtype for instance, dtype in device_instance_types}

    # Parse the device values using the previously defined function
    device_values = parse_instance_properties(filepath, device_info)

    # Adding device type information to the parsed values
    for instance_name, instance_values in device_values.items():
        instance_values['device_type'] = instance_type_dict.get(instance_name, "Unknown")

    return device_values


def findDCValue(filepath):
    # Extract the properties using the previously defined function
    instance_properties = extract_instance_properties(filepath)

    instance_name = "V0"
    property_name = "pwr"

    # Retrieve the specific instance details
    instance_details = instance_properties.get(instance_name, None)
    if not instance_details:
        raise ValueError(f"Instance '{instance_name}' not found in the file.")

    # Retrieve the specific property value
    property_value = instance_details.get(property_name, None)
    if property_value is None:
        raise ValueError(f"Property '{property_name}' not found for instance '{instance_name}'.")

    # Convert the value to float
    try:
        float_value = float(property_value)
        float_value = abs(float_value)
        if float_value != float_value:  # Check for NaN values
            raise ValueError(f"The value of property '{property_name}' for instance '{instance_name}' is NaN.")
        return {property_name: float_value}
    except ValueError:
        raise ValueError(
            f"The value of property '{property_name}' for instance '{instance_name}' is not a valid number.")


# device_instance_properties = extractInstanceProperties("DC.raw/dcOpInfo.info.encode")
# var = list(device_instance_properties.items())[:]
# print(var)

# Testing the function of get_instance_property_value
# power = findDCValue("DC.raw/dcOpInfo.info.encode", "V0", "pwr")
# print(power)