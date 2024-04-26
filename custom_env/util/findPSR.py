from util.extract_trace import extractACTrace
# from extract_trace import extractACTrace
from util.util_func import find_closest_value_index
# from util_func import find_closest_value_index
import math


def findPowerSupplyRejectionRatio(file_path):

    freq_name = 'freq'
    vout_name = 'VOUT'

    freq_1 = 100000  # 100kHz
    freq_2 = 1000000  # 1MHz
    freq_3 = 10000000  # 10MHz

    try:
        trace_dict = extractACTrace(file_path)

        freq_trace = trace_dict[freq_name]
        vout_trace = trace_dict[vout_name]

        index_freq_1 = find_closest_value_index(freq_trace, freq_1)
        index_freq_2 = find_closest_value_index(freq_trace, freq_2)
        index_freq_3 = find_closest_value_index(freq_trace, freq_3)

        psr_1 = -20 * math.log10(vout_trace[index_freq_1])
        psr_2 = -20 * math.log10(vout_trace[index_freq_2])
        psr_3 = -20 * math.log10(vout_trace[index_freq_3])

        if psr_1 <= 0.0:
            psr_1 = 0.0
            print("Warning: PSR at 100kHz is positive")
        if psr_2 <= 0.0:
            psr_2 = 0.0
            print("Warning: PSR at 1MHz is positive")
        if psr_3 <= 0.0:
            psr_3 = 0.0
            print("Warning: PSR at 10MHz is positive")

    except Exception as e:
        print("Warning: Extract PSR error", e)
        psr_1 = 0.0
        psr_2 = 0.0
        psr_3 = 0.0

    return {"psr_100k": psr_1, "psr_1M": psr_2, "psr_10M": psr_3}


# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio("/Users/hanwu/Downloads/ac.ac.encode")
# print(value_dict)
