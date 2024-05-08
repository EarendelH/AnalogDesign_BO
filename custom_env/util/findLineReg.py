from util.extract_trace import extractTrace
# from extract_trace import extractTrace
import os
import matplotlib.pyplot as plt


def findLineReg(filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    vdd_name = '"VIN"'
    vout_name = '"VOUT"'

    trace_dict = extractTrace(filename)

    vdd_trace = trace_dict[vdd_name]
    vout_trace = trace_dict[vout_name]

    vdd_init = vdd_trace[0]
    vdd_end = vdd_trace[-1]
    vout_init = vout_trace[0]
    vout_end = vout_trace[-1]

    try:
        line_reg = (vout_end - vout_init) / (vdd_end - vdd_init)
        line_reg = abs(line_reg)
    except ZeroDivisionError:
        line_reg = 100.0
        print("Warning: Division by zero. Setting load regulation to 100.0.")

    return {"lineReg": line_reg}

# Test Code
# file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Line_Reg_500u.raw/dc.dc.encode"
# print(findLineReg(file))


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'Line_Reg_200m.raw', 'dc.dc.encode')
        if os.path.isfile(target_file):
            trace_data = extractTrace(target_file)
            print(trace_data)
            results[subdir] = {
                'VDD': trace_data['"VDD"'],
                'VOUT': trace_data['"VOUT"']
            }

    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            plt.plot(data['VDD'][0::2], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            # plt.plot(data['VDD'][0::2], data['VOUT'], label=subdir, color='blue', linewidth=1)
            plt.plot(data['VDD'][0::2], data['VOUT'], label=subdir, linewidth=1)

    plt.title('VOUT vs. VDD Plot, Iload = 200mA')
    plt.xlabel('VDD')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'Line_Reg_200m.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


# root_dir = '/Users/hanwu/Downloads/Joblib/SSF_UM_fc0a3/fm_sort'
# highlight_subdir = 'tmp_20240416222829162161766'
# scan_and_process(root_dir, highlight_subdir)
