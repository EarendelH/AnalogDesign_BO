import re
import os
import subprocess
import numpy as np
from extract_bode import analyze_loop_gain_only_value
# from util.extract_bode import analyze_loop_gain_only_value


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


def findPhaseMarginAndGBW_Jianping(encoded_file_path):
    """
    Custom function to analyze phase margin and gain bandwidth with modified phase checks:
    1. Check phases at specific frequencies (10kHz, 100kHz, 1MHz)
    2. Find phase at lowest frequency
    3. Calculate phase differences and check if less than 80 degrees

    Parameters:
    encoded_file_path: Absolute path to the stb.margin.stb.encode file

    Returns:
    dict: {"phaseMargin": float, "gainBandWidth": float}
        Returns zeros if phase requirements are not met
    """
    try:
        # Extract directory path from the encoded file path
        directory_path = os.path.dirname(encoded_file_path)

        # Construct path for stb.stb file in the same directory
        file_to_process = os.path.join(directory_path, "stb.stb")
        if not os.path.exists(file_to_process):
            print(f"Warning!!! stb.stb not found in {directory_path}")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Generate processed file path
        processed_file = os.path.join(directory_path, "stb.stb.encode")

        # Run psf command
        try:
            subprocess.run(f"psf {file_to_process} -o {processed_file}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning!!! Error running psf command: {e}")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Check if processed file was created
        if not os.path.exists(processed_file):
            print(f"Warning!!! Processed file not created at {processed_file}")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Analyze loop gain using extract_bode function
        df = analyze_loop_gain_only_value(processed_file)

        # Step 1: Get phases at specific frequencies
        test_freqs = [1e4, 1e5, 1e6]
        specific_phases = []

        for test_freq in test_freqs:
            freq_array = df['Frequency (Hz)'].values
            closest_idx = np.argmin(np.abs(freq_array - test_freq))
            actual_freq = freq_array[closest_idx]
            phase = df.iloc[closest_idx]['Phase (degrees)']
            specific_phases.append(phase)
            print(f"Phase at {actual_freq:.2e} Hz: {phase:.2f} degrees")

        # Step 2: Get phase at lowest frequency
        lowest_freq_idx = np.argmin(df['Frequency (Hz)'].values)
        lowest_freq_phase = df.iloc[lowest_freq_idx]['Phase (degrees)']
        lowest_freq = df.iloc[lowest_freq_idx]['Frequency (Hz)']
        print(f"Phase at lowest frequency ({lowest_freq:.2e} Hz): {lowest_freq_phase:.2f} degrees")

        # Step 3: Check phase differences
        for i, specific_phase in enumerate(specific_phases):
            phase_difference = abs(lowest_freq_phase - specific_phase)
            print(f"Phase difference with {test_freqs[i]:.2e} Hz: {phase_difference:.2f} degrees")

            if phase_difference < 80:
                print(f"Warning!!! Phase difference requirement not met: {phase_difference:.2f} degrees < 80 degrees")
                return {"phaseMargin": 0, "gainBandWidth": 0}

        # If all phase checks pass, use original findPhaseMarginAndGBW function
        print("All phase requirements met!")
        return findPhaseMarginAndGBW(encoded_file_path)

    except Exception as e:
        print(f"Warning!!! Error in findPhaseMarginAndGBW_Jianping: {e}")
        return {"phaseMargin": 0, "gainBandWidth": 0}


# Test the function with the provided file
value_dict = findPhaseMarginAndGBW_Jianping("/home/wuhan/Downloads/Test_Netlist/Bad_Stb.raw/stb.margin.stb.encode")
print(f"New util: {value_dict}")

value_dict = findPhaseMarginAndGBW("/home/wuhan/Downloads/Test_Netlist/Bad_Stb.raw/stb.margin.stb.encode")
print(f"Old util: {value_dict}")
