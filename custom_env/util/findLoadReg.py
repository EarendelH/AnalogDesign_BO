from util.extract_trace import extractTrace


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
        load_reg = 100.0
        print("Warning: Division by zero. Setting load regulation to 100.0.")

    return {"loadReg": load_reg}

# Test Code
# file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Load_Reg.raw/dc.dc.encode"
# print(findLoadReg(file))
