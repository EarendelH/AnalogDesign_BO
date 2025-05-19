import re

def update_multi_values(file1, file2, output):
    pattern_line = re.compile(r'^\s*M(\d{1,3})\b.*multi=(\d+)')
    # 第一步\: 从file1读取并记录multi值
    multi_map = {}
    with open(file1, 'r') as f1:
        for line in f1:
            match = pattern_line.search(line)
            if match:
                device = match.group(1)
                multi_val = match.group(2)
                multi_map[device] = multi_val

    unmatched = set()
    lines_out = []
    # 第二步\: 在file2中更新multi值
    with open(file2, 'r') as f2:
        for line in f2:
            match = pattern_line.search(line)
            if match:
                device = match.group(1)
                if device in multi_map:
                    old_str = f'multi={match.group(2)}'
                    new_str = f'multi={multi_map[device]}'
                    line = line.replace(old_str, new_str, 1)
                else:
                    unmatched.add(device)
            lines_out.append(line)

    # 第三步\: 打印未匹配设备并写出结果
    for dev in unmatched:
        print(f'警告\: 未找到匹配的multi值 -> M{dev}')
    with open(output, 'w') as fo:
        fo.writelines(lines_out)

# 示例用法
# update_multi_values('file1.scs', 'file2.scs', 'output.scs')

if __name__ == "__main__":
    original_netlist = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_original/netlist_original_LFM/DC.scs"
    updated_netlist = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_LFM/DC_parameterized.scs"

    output_netlist = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_LFM/DC_parameterized_updated.scs"
    update_multi_values(original_netlist, updated_netlist, output_netlist)