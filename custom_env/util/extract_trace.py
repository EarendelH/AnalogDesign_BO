from collections import defaultdict
import re


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


def extractTransTrace(filepath):
    """
    Extract the trace data from a file.

    Args:
    - file: File object to be processed.
    """

    with open(filepath, "r") as filepath:
        content = filepath.readlines()
    # Find TRACE, VALUE and END lines, make segments

    is_collecting = False
    extracted_trace_lines = []

    for line in content:
        if "TRACE" in line:
            is_collecting = True
            continue
        elif "VALUE" in line:
            break
        if is_collecting:
            extracted_trace_lines.append(line)

    is_collecting = False
    extracted_trace_lines_refine = []

    for line in extracted_trace_lines:
        if ")" in line:
            is_collecting = True
            continue
        if is_collecting:
            extracted_trace_lines_refine.append(line)

    is_collecting = False
    extracted_value_lines = []

    for line in content:
        if "VALUE" in line:
            is_collecting = True
            continue
        if "END" in line:
            break
        if is_collecting:
            extracted_value_lines.append(line)

    trace_name_list = []
    for line in extracted_trace_lines_refine:
        parts = line.split()
        trace_name = parts[0].strip('"')
        trace_name_list.append(trace_name)

    # Separate extracted_value_lines by "time"
    extracted_trace = defaultdict(list)
    line_count = 0
    for line in extracted_value_lines:
        if line.startswith('"time"'):
            cur_time = float(line.split()[1])
            extracted_trace["time"].append(cur_time)
            line_count = 0
        elif line.startswith('"group"'):
            continue
        else:
            trace_key = trace_name_list[line_count % len(trace_name_list)]
            line_count += 1
            extracted_trace[trace_key].append(float(line))

    trace_dict = dict(extracted_trace)

    return trace_dict


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
