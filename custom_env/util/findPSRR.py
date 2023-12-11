import re
import math


def findPowerSupplyRejectionRatio(file_path):
    # Initialize the flags and storage variables.
    found_value = False
    net03_value = None
    net3_value = None

    # Regular expression pattern to find the floating point numbers.
    pattern = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")

    # Read the file and iterate over its lines
    with open(file_path, "r") as file:
        for line in file:
            # Check if the line contains "VALUE"
            if "VALUE" in line:
                found_value = True
                continue

            # If "VALUE" has been found, then look for "net03" and "net3" lines
            if found_value:
                # Check for net03 line and extract the first floating point number
                if "net03" in line and net03_value is None:
                    net03_values = re.findall(pattern, line)
                    net03_value = float(net03_values[1][0] + (net03_values[1][2] if net03_values[1][2] else ""))
                # Check for net3 line and extract the first floating point number
                elif "net3" in line and net3_value is None:
                    net3_values = re.findall(pattern, line)
                    net3_value = float(net3_values[1][0] + (net3_values[1][2] if net3_values[1][2] else ""))

            # Break the loop if both values have been found.
            if net03_value is not None and net3_value is not None:
                break

    # Calculate the ratio and return the result
    power_supply_rejection_ratio = net3_value / net03_value
    power_supply_rejection_ratio = 20 * math.log10(power_supply_rejection_ratio)
    power_supply_rejection_ratio = abs(power_supply_rejection_ratio)
    return {"powerSupplyRejectionRatio": power_supply_rejection_ratio}


# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio("PSRR.raw/ac.ac.encode")
# print(value_dict)
