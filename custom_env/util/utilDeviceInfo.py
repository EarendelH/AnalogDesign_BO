import re
import csv


def parseDeviceInfo(filepath):
    def extract_device_properties(device_section):
        # Extract properties for each device with a flexible approach
        property_patterns = [
            (r'\"(.+?)\" FLOAT DOUBLE PROP\(\n(\"units\" \"(.+?)\"\n)?\"description\" \"(.+?)\"\n\)', 'FLOAT DOUBLE'),
            (r'\"(.+?)\" INT BYTE PROP\(\n\"description\" \"(.+?)\"\n\)', 'INT BYTE')
        ]

        properties = []
        for pattern, dtype in property_patterns:
            property_matches = re.findall(pattern, device_section)
            for match in property_matches:
                if dtype == 'FLOAT DOUBLE':
                    prop_name, _, prop_unit, prop_description = match
                elif dtype == 'INT BYTE':
                    prop_name, prop_description = match
                    prop_unit = None
                property_info = {
                    'property_name': prop_name.strip(),
                    'property_type': dtype,
                    'property_unit': prop_unit.strip() if prop_unit else None,
                    'property_description': prop_description.strip()
                }
                properties.append(property_info)

        return properties

    # Read the file content
    with open(filepath, "r") as file:
        file_content = file.read()

    # Check if the required section exists
    match = re.search(r'TYPE(.+?)VALUE', file_content, re.DOTALL)
    if not match:
        raise ValueError("The content between 'TYPE' and 'VALUE' not found in the file.")

    type_value_section = match.group(1)

    # Split the section into individual device sections based on the provided delimiter
    device_sections = type_value_section.split('PROP(\n"key" "inst"\n)\n')

    # Filter out empty or invalid sections and parse each device section
    parsed_devices = []
    for section in device_sections:
        if section.strip() and '" STRUCT(' in section:
            device_name_match = re.search(r'\"(\w+)\" STRUCT', section)
            device_name = device_name_match.group(1) if device_name_match else None
            properties = extract_device_properties(section)
            device_info = {
                'device_name': device_name,
                'device_type': 'STRUCT',
                'properties': properties
            }
            parsed_devices.append(device_info)
    return parsed_devices


def deviceInfoExport(device_info, output_filepath):
    # CSV headers
    headers = ["DeviceName", "DeviceType", "ParameterName", "ParameterDateType", "ParameterUnit",
               "ParameterDescription"]

    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for device in device_info:
            device_name = device['device_name']
            device_type = device['device_type']
            for prop in device['properties']:
                row = [
                    device_name,
                    device_type,
                    prop['property_name'],
                    prop['property_type'],
                    prop['property_unit'],
                    prop['property_description']
                ]
                writer.writerow(row)


# Testing the function with the provided content
# test_input_filepath = "/Users/hanwu/ML/PSF_Read/DC.raw/dcOpInfo.info.encode"
# test_output_filepath = "/Users/hanwu/ML/PSF_Read/DC.raw/dcOpInfo.info.encode.csv"

# test_parsed_devices_info = parseDeviceInfo(test_input_filepath)
# deviceInfoExport(test_parsed_devices_info, test_output_filepath)
