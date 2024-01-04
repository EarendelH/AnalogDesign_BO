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
        lines = file.readlines()

        phase_margin = None
        phase_margin_frequency = None

        # Extract phaseMargin value
        for line in lines:
            if '"phaseMargin" "Deg"' in line:
                phase_margin = float(line.split()[2])
            if '"phaseMarginFreq" "Hz"' in line:
                phase_margin_frequency = float(line.split()[2])

            # Break early if both values have been found
            if phase_margin is not None and phase_margin_frequency is not None:
                break

            # Set to default values if not found
            if phase_margin is None or phase_margin_frequency is None:
                phase_margin = 0.1
                phase_margin_frequency = 0.1
                print("Warning: Unstable, phaseMargin or phaseMarginFreq not Existing, set to 0.1")

        # Calculate gainBandWidth
        gain_bandwidth = phase_margin * phase_margin_frequency

        return {"phaseMargin": phase_margin, "gainBandWidth": gain_bandwidth}

# Test the function with the provided file
# value_dict = findPhaseMarginAndGBW("/Users/hanwu/ML/AnalogDesignAuto/resultParse/spectreEnv/spectreTmpFile/tmp_202308301525373103/Stability.raw/stb.margin.stb.encode")
# print(value_dict)

# value_dict = findLoopGain("/Users/hanwu/ML/AnalogDesignAuto/resultParse/spectreEnv/spectreTmpFile/tmp_202308211608431287/Stability.raw/stb.stb.encode")
# print(value_dict)
