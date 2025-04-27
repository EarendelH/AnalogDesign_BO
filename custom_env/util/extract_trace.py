import re
from collections import defaultdict


def extractTrace(filepath):
    """
    Extract the trace data from a file.

    Args:
    - file: File object to be processed.
    """

    is_collecting = False
    extracted_lines = []

    with open(filepath, 'r') as filepath:
        lines = filepath.readlines()
    # Find TRACE line and END line
    for line in lines:
        if "VALUE" in line:
            is_collecting = True
            continue
        elif "END" in line:
            break
        if is_collecting:
            extracted_lines.append(line)

    extracted_trace = defaultdict(list)
    for line in extracted_lines:
        parts = line.split()
        trace_name = parts[0]
        trace_value = float(parts[1])
        extracted_trace[trace_name].append(trace_value)

    trace_dict = dict(extracted_trace)

    return trace_dict


def extractTransTrace(file_path):
    """
    Processes the signal file to extract time series data for each signal, correctly handling the 'group' line.

    Args:
    file_path (str): Path to the signal file.

    Returns:
    dict: A dictionary containing time series data for each signal.
    """
    # Read the entire file
    with open(file_path, 'r') as file:
        all_lines = file.readlines()

    # Find indices for "TRACE", "VALUE", and "END"
    trace_index = next(i for i, line in enumerate(all_lines) if "TRACE" in line)
    value_index = next(i for i, line in enumerate(all_lines) if "VALUE" in line)
    end_index = next(i for i, line in enumerate(all_lines) if "END" in line)

    # Extract lines between "TRACE" and "VALUE" for signal names
    trace_name_lines = all_lines[trace_index + 1:value_index]
    # print(trace_name_lines)

    # Remove lines from "PROP(" ending line to the nearest ")" line
    filtered_lines = []
    skip = False

    for line in trace_name_lines:
        if skip:
            if ")" in line:
                skip = False
        else:
            if line.strip().endswith("PROP("):
                filtered_lines.append(line)
                skip = True
            else:
                filtered_lines.append(line)

    # print(filtered_lines)

    extracted_signal_name = []
    for line in filtered_lines:
        match = re.search(r'\"(.*?)\"', line)
        if match:
            extracted_signal_name.append(match.group(1))
    extracted_signal_name.pop(0)
    # print(extracted_signal_name)

    trace_value_lines = all_lines[value_index + 1:end_index]
    # print(trace_value_lines)

    grouped_trace_values = []
    current_group = []

    for line in trace_value_lines:
        if line.strip().startswith('"time"'):
            if current_group:
                grouped_trace_values.append(current_group)
            current_group = [line]
        else:
            current_group.append(line)

    # Add the last group if it's not empty
    if current_group:
        grouped_trace_values.append(current_group)
    # print(grouped_trace_values)

    for group in grouped_trace_values:
        for i, line in enumerate(group):
            if line.strip().startswith('"group"'):
                group[i] = line.replace('"group"', '').strip()
    # print(grouped_trace_values)

    signal_data = {name: [] for name in extracted_signal_name}
    signal_data["time"] = []

    # Process each group in grouped_trace_values
    for group in grouped_trace_values:
        # Find the "time" line and extract the timestamp
        time_line = next(line for line in group if line.strip().startswith('"time"'))
        timestamp = float(time_line.replace('"time"', '').strip())

        # Process each line after the "time" line
        for line, name in zip(group[group.index(time_line) + 1:], extracted_signal_name):
            # Add the signal data to the corresponding list in signal_data
            signal_data[name].append(float(line.strip()))

        # Add the timestamp to the "time" list
        signal_data["time"].append(timestamp)

    # print(signal_data)

    return signal_data

# Test Code
# file = "/Users/hanwu/Downloads/tb_Efficiency_1.raw/tran.tran.tran.encode"
# dict = extractTransTrace(file)
# # Print signal name
# print(dict.keys())
# # Print time series data
# print(dict["time"])


def extractTransTrace_psf(file_path):
    """
    Extract transient simulation data from PSF format file with improved handling of scientific notation
    and structured data organization.

    Args:
        file_path (str): Path to the PSF format file

    Returns:
        dict: A dictionary containing the time series data for each signal
            Format: {
                'time': [t1, t2, t3, ...],
                'signal1': [v1, v2, v3, ...],
                'signal2': [v1, v2, v3, ...],
                ...
            }
    """
    # Initialize storage for signals
    signals_data = {}
    current_time = None

    try:
        with open(file_path, 'r') as file:
            # Find VALUE and END section
            content = file.read()
            value_start = content.find("VALUE")
            end_pos = content.find("END")

            if value_start == -1 or end_pos == -1:
                raise ValueError("Invalid file format: VALUE or END section not found")

            # Extract data section
            data_section = content[value_start:end_pos].split('\n')

            # Process each line
            for line in data_section:
                line = line.strip()
                if not line:
                    continue

                # Extract signal name and value using regex
                match = re.match(r'"([^"]+)"\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', line)
                if not match:
                    continue

                signal_name, value_str = match.groups()

                # Convert value to float
                try:
                    value = float(value_str)
                except ValueError:
                    print(f"Warning: Could not convert value {value_str} for signal {signal_name}")
                    continue

                # Handle time points
                if signal_name == "time":
                    current_time = value
                    if 'time' not in signals_data:
                        signals_data['time'] = []
                    signals_data['time'].append(value)
                else:
                    # Initialize list for new signals
                    if signal_name not in signals_data:
                        signals_data[signal_name] = []
                    signals_data[signal_name].append(value)

        # Validate data consistency
        expected_length = len(signals_data['time'])
        for signal_name, values in signals_data.items():
            if len(values) != expected_length:
                print(f"Warning: Signal {signal_name} has inconsistent length")
                # Pad with None or handle inconsistency as needed
                while len(values) < expected_length:
                    values.append(None)

        return signals_data

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return {}
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {}


# Test Code
# file = "/Users/hanwu/Downloads/tb_Efficiency/tb_Efficiency.raw/tran.tran.tran"
# demo_dict = extractTransTrace_psf(file)
# Print signal name
# print(demo_dict.keys())
# Print time series data
# print(f"The raw series length is {len(demo_dict['time'])}")
# Filter time and SW_IDEAL signal within time 298.5u-300us, and write to csv file
# import csv
# with open('demo.csv', mode='w') as csv_file:
#     fieldnames = ['time', 'SW_IDEAL']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     for i in range(len(demo_dict["time"])):
#         if 295e-6 <= demo_dict["time"][i] <= 300e-6:
#             writer.writerow({'time': demo_dict["time"][i], 'SW_IDEAL': demo_dict["SW_IDEAL"][i]})
# Print the length of raw series length and filtered series length


def extractACTrace(file_path):
    # Function to read content from a file
    def read_file_content(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    # Function to extract values between "VALUE" and "END"
    def find_value_content(content):
        start = content.index("VALUE") + len("VALUE")
        end = content.index("END")
        return content[start:end]

    # Use regex to parse the signal names and their corresponding values
    def parse_signals(value_content):
        # Updated regex to better handle numbers, including scientific notation
        regex = r'\"(.+?)\"\s+(\(.*?\)|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        matches = re.findall(regex, value_content)
        signal_data = {}
        for match in matches:
            signal_name, value_str = match
            try:
                if value_str.startswith('('):
                    # Extract the first float from a pair of floats in parentheses, ensuring full scientific notation
                    # is captured
                    first_value, second_value = re.match(r'\((-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)[ ,]+(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)', value_str).groups()
                    first_value = float(first_value)
                    second_value = float(second_value)
                    value = (first_value * first_value + second_value * second_value) ** 0.5
                else:
                    # Directly convert the string to float
                    value = float(value_str)
                if signal_name not in signal_data:
                    signal_data[signal_name] = []
                signal_data[signal_name].append(value)
            except ValueError as e:
                print(f"Error converting {value_str} to float: {e}")
                continue
        return signal_data

    # Read content from file
    content = read_file_content(file_path)

    # Extract and parse the signal values
    value_content = find_value_content(content)
    signal_data = parse_signals(value_content)
    return signal_data


# Test Code
# file = "/Users/hanwu/Downloads/ac.ac.encode"
# dict = extractACTrace(file)
# print(dict)

def generate_trans_signals_csv(file_path):
    """
    Generate a CSV file containing all signals from a PSF format file.
    The first column is the unified timestamp, followed by all signal values.
    The CSV file is saved in the same directory as the PSF file.

    生成包含PSF格式文件中所有信号的CSV文件。
    第一列是统一的时间戳，之后是所有信号值。
    CSV文件保存在与PSF文件相同的目录下。

    Args:
        file_path (str): Path to the PSF format file
                        PSF格式文件的路径

    Returns:
        str: Path to the generated CSV file, or None if an error occurred
             生成的CSV文件路径，如果发生错误则返回None
    """
    import os
    import csv

    try:
        # Extract all signal data
        # 提取所有信号数据
        signal_dict = extractTransTrace_psf(file_path)

        # Check if data extraction was successful
        # 检查数据提取是否成功
        if not signal_dict or "time" not in signal_dict:
            print(f"Error: Failed to extract data from {file_path} or time series does not exist")
            return None

        # Get time series
        # 获取时间序列
        time_series = signal_dict["time"]

        # Get all signal names (excluding time)
        # 获取所有信号名称（不包括时间）
        signal_names = [name for name in signal_dict.keys() if name != "time"]
        print(f"Extracted {len(signal_names)} signals from {file_path}")

        # Determine output file path (same directory as input file)
        # 确定输出文件路径（与输入文件相同目录）
        output_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path).split('.')[0]
        output_file = os.path.join(output_dir, f"{base_name}_signals.csv")

        # Create CSV file
        # 创建CSV文件
        with open(output_file, 'w', newline='') as csvfile:
            # Create CSV writer
            # 创建CSV写入器
            fieldnames = ["time"] + signal_names
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            # 写入表头
            writer.writeheader()

            # Write data rows
            # 写入数据行
            for i in range(len(time_series)):
                row = {"time": time_series[i]}
                for signal_name in signal_names:
                    if i < len(signal_dict[signal_name]):
                        row[signal_name] = signal_dict[signal_name][i]
                    else:
                        row[signal_name] = None
                writer.writerow(row)

        print(f"Signal data saved to: {output_file}")
        print(f"Total time points: {len(time_series)}")
        print(f"Total signals: {len(signal_names)}")

        return output_file

    except Exception as e:
        print(f"Error generating CSV file: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    file_path = "/Users/hanwu/Downloads/Trans.raw/tran.tran.tran"
    generate_trans_signals_csv(file_path)
