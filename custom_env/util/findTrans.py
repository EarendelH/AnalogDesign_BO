import numpy as np
import os
import matplotlib.pyplot as plt
# from extract_trace import extractTransTrace
from util.extract_trace import extractTransTrace
import numpy as np


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
        if signal_value[i - 1] < mid_value < signal_value[i] and signal_value[i + 1] > mid_value:
            rising_edges.append(i)
        elif signal_value[i - 1] > mid_value > signal_value[i] and signal_value[i + 1] < mid_value:
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
    slew_rate_up_val = np.mean(slew_rates_up).item() if slew_rates_up else None
    slew_rate_down_val = np.mean(slew_rates_down).item() if slew_rates_down else None

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
    if slew_rate_up_val is None:
        slew_rate_up_val = 0.0
        print(f"Warning!!! slewRateUp is not found, set to 0.0")
    if slew_rate_down_val is None:
        slew_rate_down_val = 0.0
        print(f"Warning!!! slewRateDown is not found, set to 0.0")
    if slew_rate_up_val < 0.0:
        slew_rate_up_val = 0.0
        print(f"Warning!!! slewRateUp is negative, set to 0.0")
    if slew_rate_down_val < 0.0:
        slew_rate_down_val = 0.0
        print(f"Warning!!! slewRateDown is negative, set to 0.0")

    return {"slewRateUp": slew_rate_up_val, "slewRateDown": slew_rate_down_val}


# dict = findSlewRate("/Users/hanwu/ML/AnalogDesignAuto/resultParse/spectreEnv/spectreTmpFile/
# tmp_202308181553507320/Trans.raw/tran.tran.tran.encode")
# print(dict)


def findShoot(filename):
    """
    Extract the overshoot and undershoot value from trans file

    Args:
    - filename: Path to the file to be processed.

    Returns:
    - Overshoot and undershoot value
    """

    trans_dict = extractTransTrace(filename)
    time_series = trans_dict["time"]
    vout_trace = trans_dict["VOUT"]
    stable_voltage = 1.0

    # Clip time 50us-100us and 100us-150us
    time_undershoot_index = [i for i, t in enumerate(time_series) if 2.5e-6 <= t <= 7.5e-6]
    time_overshoot_index = [i for i, t in enumerate(time_series) if 7.5e-6 <= t <= 12.5e-6]
    vout_undershoot = [vout_trace[i] for i in time_undershoot_index]
    vout_overshoot = [vout_trace[i] for i in time_overshoot_index]

    # Calculate overshoot and undershoot,
    # undershoot: Vout@50us -Vout_clip1_min, overshoot: Vout_clip2_max - Vout@100us
    vout_undershoot_min = np.min(vout_undershoot)
    vout_overshoot_max = np.max(vout_overshoot)
    # Find the neset value to 50us and 100us
    vout_undershoot_base = vout_undershoot[0]
    vout_overshoot_base = vout_overshoot[0]

    undershoot = vout_undershoot_base - vout_undershoot_min
    overshoot = vout_overshoot_max - vout_overshoot_base

    # Determine whether the stable voltage is regulated to 1.2V (pre-defined)
    # Find the mid-value in vout_undershoot
    stable_high_load_voltage = vout_undershoot[-1]
    stable_light_load_voltage = vout_overshoot[-1]
    # print(f"Debug, stable_light_load_voltage: {stable_light_load_voltage}")
    if stable_high_load_voltage >= stable_voltage * 1.1 or stable_high_load_voltage <= stable_voltage * 0.9:
        print("Warning! This LDO cannot be regulated to VREF under high load.")
        overshoot = 100.0
        undershoot = 1.0
    if stable_light_load_voltage >= stable_voltage * 1.1 or stable_light_load_voltage <= stable_voltage * 0.9:
        print("Warning! This LDO cannot be regulated to VREF under light load.")
        overshoot = 100.0
        undershoot = 100.0
    else:
        pass

    return {"overShoot": overshoot, "underShoot": undershoot}

# Test Code
# file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Trans.raw/tran.tran.tran.encode"
# print(findShoot(file))

# file = "/Users/hanwu/Downloads/Joblib/SSF_UM_53c63/tmp_20240411224832550072069/Trans.raw/tran.tran.tran.encode"
# print(findShoot(file))


def findShoot_Jiangping(filename):
    """
    Extract the overshoot and undershoot value from trans file

    Args:
    - filename: Path to the file to be processed.

    Returns:
    - Overshoot and undershoot value
    """

    trans_dict = extractTransTrace(filename)
    time_series = trans_dict["time"]
    vout_trace = trans_dict["VOUT"]
    stable_voltage = 0.5

    # Clip time 50us-100us and 100us-150us
    time_undershoot_index = [i for i, t in enumerate(time_series) if 2.5e-6 <= t <= 7.5e-6]
    time_overshoot_index = [i for i, t in enumerate(time_series) if 7.5e-6 <= t <= 12.5e-6]
    vout_undershoot = [vout_trace[i] for i in time_undershoot_index]
    vout_overshoot = [vout_trace[i] for i in time_overshoot_index]

    # Calculate overshoot and undershoot,
    # undershoot: Vout@50us -Vout_clip1_min, overshoot: Vout_clip2_max - Vout@100us
    vout_undershoot_min = np.min(vout_undershoot)
    vout_overshoot_max = np.max(vout_overshoot)
    # Find the neset value to 50us and 100us
    vout_undershoot_base = vout_undershoot[0]
    vout_overshoot_base = vout_overshoot[0]

    undershoot = vout_undershoot_base - vout_undershoot_min
    overshoot = vout_overshoot_max - vout_overshoot_base

    # Determine whether the stable voltage is regulated to 1.2V (pre-defined)
    # Find the mid-value in vout_undershoot
    stable_high_load_voltage = vout_undershoot[-1]
    stable_light_load_voltage = vout_overshoot[-1]
    # print(f"Debug, stable_light_load_voltage: {stable_light_load_voltage}")
    if stable_high_load_voltage >= stable_voltage * 1.1 or stable_high_load_voltage <= stable_voltage * 0.9:
        print("Warning! This LDO cannot be regulated to VREF under high load.")
        overshoot = 100.0
        undershoot = 1.0
    if stable_light_load_voltage >= stable_voltage * 1.1 or stable_light_load_voltage <= stable_voltage * 0.9:
        print("Warning! This LDO cannot be regulated to VREF under light load.")
        overshoot = 100.0
        undershoot = 100.0
    else:
        pass

    return {"overShoot": overshoot, "underShoot": undershoot}


def findShoot_Line_Reg(filename):
    """
    Extract the overshoot and undershoot value from trans file

    Args:
    - filename: Path to the file to be processed.

    Returns:
    - Overshoot and undershoot value
    """

    trans_dict = extractTransTrace(filename)
    time_series = trans_dict["time"]
    vout_trace = trans_dict["VOUT"]
    stable_voltage = 0.5

    # Clip time 50us-100us and 100us-150us
    time_undershoot_index = [i for i, t in enumerate(time_series) if 320.0e-6 <= t <= 400.0e-6]
    time_overshoot_index = [i for i, t in enumerate(time_series) if 60.0e-6 <= t <= 140.0e-6]
    vout_undershoot = [vout_trace[i] for i in time_undershoot_index]
    vout_overshoot = [vout_trace[i] for i in time_overshoot_index]

    # Calculate overshoot and undershoot,
    # undershoot: Vout@50us -Vout_clip1_min, overshoot: Vout_clip2_max - Vout@100us
    vout_undershoot_min = np.min(vout_undershoot)
    vout_overshoot_max = np.max(vout_overshoot)
    # Find the neset value to 50us and 100us
    vout_undershoot_base = vout_undershoot[0]
    vout_overshoot_base = vout_overshoot[0]

    undershoot = vout_undershoot_base - vout_undershoot_min
    overshoot = vout_overshoot_max - vout_overshoot_base

    # Determine whether the stable voltage is regulated to 1.2V (pre-defined)
    # Find the mid-value in vout_undershoot
    stable_high_load_voltage = vout_undershoot[-1]
    stable_light_load_voltage = vout_overshoot[-1]
    # print(f"Debug, stable_light_load_voltage: {stable_light_load_voltage}")
    if stable_high_load_voltage >= stable_voltage * 1.1 or stable_high_load_voltage <= stable_voltage * 0.9:
        print("Warning! This LDO cannot be regulated to VREF under high load.")
        overshoot = 100.0
        undershoot = 1.0
    if stable_light_load_voltage >= stable_voltage * 1.1 or stable_light_load_voltage <= stable_voltage * 0.9:
        print("Warning! This LDO cannot be regulated to VREF under light load.")
        overshoot = 100.0
        undershoot = 100.0
    else:
        pass

    return {"overShoot": overshoot, "underShoot": undershoot}


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'Trans.raw', 'tran.tran.tran.encode')
        if os.path.isfile(target_file):
            trace_data = extractTransTrace(target_file)
            results[subdir] = {
                'time': trace_data['time'],
                'VOUT': trace_data['VOUT']
            }
            print(trace_data['time'])
            print(trace_data['VOUT'])
    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            plt.plot(data['time'], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            # plt.plot(data['time'], data['VOUT'], label=subdir, color='blue', linewidth=1)
            plt.plot(data['time'], data['VOUT'], label=subdir, linewidth=1)

    plt.title('VOUT vs. Time Plot, Iload = 200mA')
    plt.xlabel('Time')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'vout_time_plot.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


# root_dir = '/Users/hanwu/Downloads/Joblib/SSF_UM_fc0a3/fm_sort'
# highlight_subdir = 'tmp_20240416222829162161766'
# scan_and_process(root_dir, highlight_subdir)


def fourier_analysis(time_data, signal_data):
    sample_intervals = np.diff(time_data)
    sample_rate = 1 / np.mean(sample_intervals)

    fft_result = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(len(signal_data), 1 / sample_rate)

    positive_freqs = fft_freq > 0
    fft_magnitude = np.abs(fft_result[positive_freqs])
    fft_frequencies = fft_freq[positive_freqs]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_data, signal_data)
    plt.title('Time Series')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.stem(fft_frequencies, fft_magnitude, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sample_rate / 2)

    plt.tight_layout()
    plt.show()


# dict = extractTransTrace("/Users/hanwu/Downloads/Joblib/SSF_UM/tmp_20240411051043339987954/
# Trans.raw/tran.tran.tran.encode")
# print(dict)
# time_undershoot_index = [i for i, t in enumerate(dict['time']) if 5e-6 <= t <= 11e-6]
# time_undershoot = [dict['time'][i] for i in time_undershoot_index]
# vout_undershoot = [dict['VOUT'][i] for i in time_undershoot_index]
# fourier_analysis(time_undershoot, vout_undershoot)
