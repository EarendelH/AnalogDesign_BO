# from util.extract_trace import extractTrace
from extract_trace import extractTrace


def findLineReg(filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    vdd_name = '"VDD"'
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
        line_reg = 0.0
        print("Warning: Division by zero. Setting load regulation to 0.0.")

    return {"lineReg": line_reg}

# Test Code
# file = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/Line_Reg.raw/dc.dc.encode"
# print(findLineReg(file))
