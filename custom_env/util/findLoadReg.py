# from util.extract_trace import extractTrace
from extract_trace import extractTrace
import os
import matplotlib.pyplot as plt

def findLoadReg(filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    iload_name = '"ILOAD"'
    vout_name = '"VOUT"'

    trace_dict = extractTrace(filename)

    iload_trace = trace_dict[iload_name]
    vout_trace = trace_dict[vout_name]

    iload_init = iload_trace[0]
    iload_end = iload_trace[-1]
    vout_init = vout_trace[0]
    vout_end = vout_trace[-1]

    try:
        load_reg = (vout_end - vout_init) / (iload_end - iload_init)
        load_reg = abs(load_reg)
    except ZeroDivisionError:
        load_reg = 0.0
        print("Warning: Division by zero. Setting load regulation to 0.0.")

    return {"loadReg": load_reg}

# Test Code
# file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Load_Reg.raw/dc.dc.encode"
# print(findLoadReg(file))


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'Load_Reg.raw', 'dc.dc.encode')
        if os.path.isfile(target_file):
            trace_data = extractTrace(target_file)
            print(trace_data)
            results[subdir] = {
                'ILOAD': trace_data['"ILOAD"'],
                'VOUT': trace_data['"VOUT"']
            }

    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            plt.plot(data['ILOAD'], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            plt.plot(data['ILOAD'], data['VOUT'], label=subdir, color='blue', linewidth=1)

    plt.title('VOUT vs. ILOAD Plot')
    plt.xlabel('ILOAD')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'load_reg.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


root_dir = '/Users/hanwu/Downloads/Joblib/CM_eex03_ea68b'
highlight_subdir = 'tmp_202404120045401172475399'
scan_and_process(root_dir, highlight_subdir)

