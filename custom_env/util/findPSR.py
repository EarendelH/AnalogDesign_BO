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


def findPowerSupplyRejectionRatio_Yan(file_path):

    freq_name = 'freq'
    vout_name = 'VOUT'

    freq_1 = 1000  # 1kHz
    freq_2 = 1000000  # 1MHz
    freq_3 = 100000000  # 100MHz
    freq_4 = 1000000000  # 1GHz
    freq_5 = 10000000000  # 10GHz

    try:
        trace_dict = extractACTrace(file_path)

        freq_trace = trace_dict[freq_name]
        vout_trace = trace_dict[vout_name]

        index_freq_1 = find_closest_value_index(freq_trace, freq_1)
        index_freq_2 = find_closest_value_index(freq_trace, freq_2)
        index_freq_3 = find_closest_value_index(freq_trace, freq_3)
        index_freq_4 = find_closest_value_index(freq_trace, freq_4)
        index_freq_5 = find_closest_value_index(freq_trace, freq_5)

        psr_1 = -20 * math.log10(vout_trace[index_freq_1])
        psr_2 = -20 * math.log10(vout_trace[index_freq_2])
        psr_3 = -20 * math.log10(vout_trace[index_freq_3])
        psr_4 = -20 * math.log10(vout_trace[index_freq_4])
        psr_5 = -20 * math.log10(vout_trace[index_freq_5])

        if psr_1 <= 0.0:
            psr_1 = 0.0
            print("Warning: PSR at 1kHz is positive")
        if psr_2 <= 0.0:
            psr_2 = 0.0
            print("Warning: PSR at 1MHz is positive")
        if psr_3 <= 0.0:
            psr_3 = 0.0
            print("Warning: PSR at 100MHz is positive")
        if psr_4 <= 0.0:
            psr_4 = 0.0
            print("Warning: PSR at 1GMHz is positive")
        if psr_5 <= 0.0:
            psr_5 = 0.0
            print("Warning: PSR at 10GHz is positive")

    except Exception as e:
        print("Warning: Extract PSR error", e)
        psr_1 = 0.0
        psr_2 = 0.0
        psr_3 = 0.0
        psr_4 = 0.0
        psr_5 = 0.0

    return {"psr_1k": psr_1, "psr_1M": psr_2, "psr_100M": psr_3, "psr_1G": psr_4, "psr_10G": psr_5}


# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio("/Users/hanwu/Downloads/ac.ac.encode")
# print(value_dict)
