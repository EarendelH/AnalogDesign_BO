import re

def findLoopGain(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

        for i in range(len(lines)):
            if "VALUE" in lines[i]:
                # Extract the value from the second line below the "VALUE" line
                value_line = lines[i + 2]
                if '"loopGain"' in value_line:
                    # Split the line using spaces and then extract the value inside the parentheses
                    value_str = value_line.split()[1].lstrip("(")
                    value_float = float(value_str)
                else:
                    value_float = -100.0
                    print("Warning: loopGain not Existing, set to -100.0")
                    return {"loopGain": value_float}


def findPhaseMarginAndGBW(filename):
    with open(filename, 'r') as file:
        file_content = file.read()

    header_type_content = re.search(r'HEADER(.*?)TYPE', file_content, re.DOTALL)
    if header_type_content:
        header_type_content = header_type_content.group(1)
    else:
        print("Warning from findPhaseMarginAndGBW: HEADER or TYPE section not found in the file.")
        return {"phaseMargin": 0.0, "gainBandWidth": 0.0}

    required_keywords = ["phaseMargin", "phaseMarginFrequency"]
    has_keywords = all(keyword in header_type_content for keyword in required_keywords)

    phase_margin = phase_margin_frequency = 0.0

    if has_keywords:
        phase_margin_match = re.search(r'"phaseMargin"\s+"([\d.+e]+)\s+Deg"', header_type_content)
        phase_margin_frequency_match = re.search(r'"phaseMarginFrequency"\s+"([\d.+e]+)\s+Hz"', header_type_content)

        if phase_margin_match and phase_margin_frequency_match:
            phase_margin = float(phase_margin_match.group(1))
            phase_margin_frequency = float(phase_margin_frequency_match.group(1))
        else:
            print("Warning: phaseMargin or phaseMarginFrequency not properly formatted, set to 0.0")
    else:
        print("Warning: phaseMargin or phaseMarginFrequency not existing, set to 0.0")

    gain_bandwidth = phase_margin_frequency

    return {"phaseMargin": phase_margin, "gainBandWidth": gain_bandwidth}



# Test the function with the provided file
# value_dict = findPhaseMarginAndGBW("/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/STB.raw/stb.margin.stb.encode")
# print(value_dict)

# value_dict = findLoopGain("/Users/hanwu/ML/AnalogDesignAuto/resultParse/spectreEnv/spectreTmpFile/tmp_202308211608431287/Stability.raw/stb.stb.encode")
# print(value_dict)
