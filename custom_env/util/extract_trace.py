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

# Test Code
# file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Trans.raw/tran.tran.tran.encode"
# dict = extractTransTrace(file)
# print(dict)
# print(dict["time"])
# print(dict["VOUT"])
