from util.extract_trace import extractACTrace
# from extract_trace import extractACTrace
from util.util_func import find_closest_value_index
# from util_func import find_closest_value_index
import math
import os
import matplotlib.pyplot as plt


def findPowerSupplyRejectionRatio_general(file_path, freq_list, freq_labels):

    freq_name = 'freq'
    vout_name = 'VOUT'

    psr_values = {}
    try:
        trace_dict = extractACTrace(file_path)

        freq_trace = trace_dict[freq_name]
        vout_trace = trace_dict[vout_name]

        for freq, label in zip(freq_list, freq_labels):
            index_freq = find_closest_value_index(freq_trace, freq)
            psr = -20 * math.log10(vout_trace[index_freq])

            if psr <= 0.0:
                psr = 0.0
                print(f"Warning: PSR at {label} is positive")

            psr_values[f"psr_{label}"] = psr

    except Exception as e:
        print("Warning: Extract PSR error", e)
        for label in freq_labels:
            psr_values[f"psr_{label}"] = 0.0

    return psr_values


def findPowerSupplyRejectionRatio(file_path):

    freq_list = [100000, 1000000, 10000000]
    freq_labels = ['100k', '1M', '10M']

    result = findPowerSupplyRejectionRatio_general(file_path, freq_list, freq_labels)

    return result


def findPowerSupplyRejectionRatio_Yan(file_path):

    freq_list = [1000, 1000000, 100000000, 1000000000, 10000000000]
    freq_labels = ['1k', '1M', '100M', '1G', '10G']

    result = findPowerSupplyRejectionRatio_general(file_path, freq_list, freq_labels)

    return result
# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio_Yan("/Users/hanwu/Downloads/tmp_20240508144031562245439/PSR.raw/ac.ac.encode")
# print(value_dict)


def findPowerSupplyRejectionRatio_Jiangping(file_path):

    freq_list = [100, 1000, 10000, 100000, 1000000]
    freq_labels = ['100', '1k', '10k', '100k', '1M']

    result = findPowerSupplyRejectionRatio_general(file_path, freq_list, freq_labels)

    return result
# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio("/Users/hanwu/Downloads/ac.ac.encode")
# print(value_dict)

def findPowerSupplyRejectionRatio_Debashis(file_path):

    freq_list = [1000, 1000000, 10000000]
    freq_labels = ['1k', '1M', '10M']

    result = findPowerSupplyRejectionRatio_general(file_path, freq_list, freq_labels)

    return result


def findPowerSupplyRejectionRatio_FC_SSF(file_path):

    freq_list = [1000, 1000000, 10000000]
    freq_labels = ['1k', '1M', '10M']

    result = findPowerSupplyRejectionRatio_general(file_path, freq_list, freq_labels)

    return result


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'PSR.raw', 'ac.ac.encode')
        if os.path.isfile(target_file):
            trace_data = extractACTrace(target_file)

            trace_data['VOUT'] = [-20 * math.log10(vout) if vout != 0 else 0 for vout in trace_data['VOUT']]
            print(trace_data)
            results[subdir] = {
                'freq': trace_data['freq'],
                'VOUT': trace_data['VOUT']
            }
            print(results[subdir])

    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            plt.semilogx(data['freq'], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            # plt.plot(data['ILOAD'], data['VOUT'], label=subdir, color='blue', linewidth=1)
            plt.semilogx(data['freq'], data['VOUT'], label=subdir, linewidth=1)

    plt.title('PSR w/ 100mA Load')
    plt.xlabel('freq')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'PSR.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


# root_dir = '/Users/hanwu/Downloads/Log/Jianping/v2/shoot_min'
# highlight_subdir = 'tmp_20240515192828337377957'
# scan_and_process(root_dir, highlight_subdir)