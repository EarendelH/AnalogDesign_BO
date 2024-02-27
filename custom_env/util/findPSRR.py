import re
import math


def findPowerSupplyRejectionRatio(file_path):
    # Initialize the flags and storage variables.
    found_value = False
    net03_value = None
    net3_value = None
    default_value = 0.0

    # Regular expression pattern to find the floating point numbers.
    pattern = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")

    # Read the file and iterate over its lines
    with open(file_path, "r") as file:
        for line in file:
            if "VALUE" in line:
                found_value = True
                continue

            if found_value:
                if "net03" in line and net03_value is None:
                    net03_values = re.findall(pattern, line)
                    net03_value = float(net03_values[1][0] + (net03_values[1][2] if net03_values[1][2] else ""))
                elif "net3" in line and net3_value is None:
                    net3_values = re.findall(pattern, line)
                    net3_value = float(net3_values[1][0] + (net3_values[1][2] if net3_values[1][2] else ""))

            if net03_value is not None and net3_value is not None:
                break

    # Check if values are properly read and calculate the ratio
    if net03_value is None or net3_value is None:
        print("Error: Unable to read net03_value or net3_value from the file. PSRR set to 0.0dB.")
        power_supply_rejection_ratio = default_value
    else:
        power_supply_rejection_ratio = net3_value / net03_value
        power_supply_rejection_ratio = 20 * math.log10(power_supply_rejection_ratio)
        power_supply_rejection_ratio = abs(power_supply_rejection_ratio)

    return {"powerSupplyRejectionRatio": power_supply_rejection_ratio}

# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio("/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/run_test
# /tmp_202401090942255147/PSRR.raw/ac.ac.encode")
# print(value_dict)
