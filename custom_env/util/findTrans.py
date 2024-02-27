import numpy as np
import matplotlib.pyplot as plt


def analyze_trans_file(file_path):
    """
    Parse the signals from the given file and return a dictionary with signal names as keys
    and their corresponding 2D numpy arrays (time and value) as values.
    """
    with open(file_path, "r") as file:
        content = file.readlines()

    # Parse the TRACE signals
    trace_signals = []
    parse_trace = False
    for line in content:
        if "TRACE" in line:
            parse_trace = True
            continue
        if parse_trace:
            if '"' in line:
                trace_signals.append(line.split('"')[1])
            else:
                break

    # Parse the VALUE section for time and signal values
    times = []
    values = []
    parse_value = False
    current_values = []
    for line in content:
        if "VALUE" in line:
            parse_value = True
            continue
        if "END" in line:
            if current_values:
                values.append(current_values)
                current_values = []
            parse_value = False
            continue
        if parse_value:
            if '"time"' in line:
                times.append(float(line.split()[-1]))
                if current_values:
                    values.append(current_values)
                    current_values = []
            elif '"' in line and not line.startswith('"time"'):
                current_values.append(float(line.split()[-1]))
            else:
                try:
                    current_values.append(float(line.strip()))
                except ValueError:
                    # Skip lines that cannot be converted to float
                    pass

    # Convert lists to numpy arrays
    times_array = np.array(times)
    values_array = np.array(values)

    # Create a dictionary with signal names as keys and their 2D numpy arrays as values
    signal_data = {}
    for idx, signal in enumerate(trace_signals[1:17]):
        combined_data = np.column_stack((times_array, values_array[:, idx]))
        signal_data[signal] = combined_data

    return signal_data


def findSlewRate(file_path):
    """
    Analyze the specified signal from the given file.

    Parameters:
    - file_path (str): Path to the file containing the signals.
    - signal_name (str): Name of the signal to be analyzed.

    Returns:
    - A plot of the specified signal against time.
    - A dictionary containing the average slew rates for rising and falling edges.
    """

    # Parse the signals
    signals = analyze_trans_file(file_path)
    # Define the output signal
    signal_name = "net3"

    # Check if the specified signal is present in the parsed data
    if signal_name not in signals:
        print(f"Signal {signal_name} not found in the file!")
        return

    # Extracting the specified signal and analyzing it
    signal_data = signals[signal_name]
    signal_time = signal_data[:, 0]
    signal_value = signal_data[:, 1]

    # Calculate the mid value
    mid_value = (np.max(signal_value) + np.min(signal_value)) / 2

    # Detect rising and falling edges
    rising_edges = []
    falling_edges = []

    for i in range(1, len(signal_value) - 1):
        if signal_value[i - 1] < mid_value and signal_value[i] > mid_value and signal_value[i + 1] > mid_value:
            rising_edges.append(i)
        elif signal_value[i - 1] > mid_value and signal_value[i] < mid_value and signal_value[i + 1] < mid_value:
            falling_edges.append(i)

    # Calculate the slew rate for each rising and falling edge
    slew_rates_up = []
    slew_rates_down = []

    for edge in rising_edges:
        start_value = mid_value + 0.1 * (np.max(signal_value) - mid_value)
        end_value = mid_value + 0.9 * (np.max(signal_value) - mid_value)
        start_indices = np.where(signal_value[edge:] < start_value)[0]
        end_indices = np.where(signal_value[edge:] > end_value)[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            continue

        start_idx = start_indices[0] + edge
        end_idx = end_indices[0] + edge

        delta_v = signal_value[end_idx] - signal_value[start_idx]
        delta_t = signal_time[end_idx] - signal_time[start_idx]

        slew_rates_up.append(delta_v / delta_t)

    for edge in falling_edges:
        start_value = mid_value + 0.1 * (mid_value - np.min(signal_value))
        end_value = mid_value - 0.9 * (mid_value - np.min(signal_value))

        start_indices = np.where(signal_value[:edge] > start_value)[0]
        end_indices = np.where(signal_value[edge:] < end_value)[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            continue

        start_idx = start_indices[-1]
        end_idx = end_indices[0] + edge

        delta_v = signal_value[start_idx] - signal_value[end_idx]
        delta_t = signal_time[end_idx] - signal_time[start_idx]

        slew_rates_down.append(delta_v / delta_t)

    # Calculate the average slew rate and store in dictionaries
    slewRateUp_val = np.mean(slew_rates_up).item() if slew_rates_up else None
    slewRateDown_val = np.mean(slew_rates_down).item() if slew_rates_down else None

    # Plot the specified signal against time
    # plt.figure(figsize=(14, 6))
    # plt.plot(signal_time, signal_value, label=signal_name, color="blue")
    # plt.axhline(y=mid_value, color='r', linestyle='--', label="Mid value")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.title(f"'{signal_name}' Signal Over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # If slewRateUp or slewRateDown is None, return 0
    if slewRateUp_val is None:
        slewRateUp_val = 0.0
        print(f"Warning!!! slewRateUp is not found, set to 0.0")
    if slewRateDown_val is None:
        slewRateDown_val = 0.0
        print(f"Warning!!! slewRateDown is not found, set to 0.0")
    if slewRateUp_val < 0.0:
        slewRateUp_val = 0.0
        print(f"Warning!!! slewRateUp is negative, set to 0.0")
    if slewRateDown_val < 0.0:
        slewRateDown_val = 0.0
        print(f"Warning!!! slewRateDown is negative, set to 0.0")

    return {"slewRateUp": slewRateUp_val, "slewRateDown": slewRateDown_val}


# dict = findSlewRate("/Users/hanwu/ML/AnalogDesignAuto/resultParse/spectreEnv/spectreTmpFile/tmp_202308181553507320/Trans.raw/tran.tran.tran.encode")
# print(dict)
