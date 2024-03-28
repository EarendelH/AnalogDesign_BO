import os
import pickle
import matplotlib.pyplot as plt


def extract_specs(specs):
    params_dict = {}
    for spec in specs:
        for key, value in spec.items():
            if key not in params_dict:
                params_dict[key] = []
            params_dict[key].append(value)
    return params_dict


def plot_specs(params_dict, title):
    num_params = len(params_dict)
    cols = (num_params + 1) // 2  # 确保至少有两行
    fig, axs = plt.subplots(2, cols, figsize=(5 * cols, 10))
    fig.suptitle(title)

    for ax, (param, values) in zip(axs.flatten(), params_dict.items()):
        ax.hist(values, bins=20, alpha=0.7, label=param)
        ax.set_title(param)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # 隐藏空白的图
    for i in range(len(params_dict), 2 * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def process_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        ideal_specs = data['initial_data']['ideal_specs']

        ideal_specs_value = {}
        for key, value in ideal_specs.items():
            ideal_specs_value[key] = value['value']

        step_len = len(data['steps_data'])

        print(f"{file_path} with {step_len} steps.")
        return ideal_specs_value, step_len
    except (EOFError, pickle.UnpicklingError, Exception) as e:
        print(f"Error processing file: {file_path}. Skipping. Error: {e}")
        return None


if __name__ == "__main__":
    dir_path = '/Users/hanwu/Downloads/run_sum'
    achieve_specs = []
    fail_specs = []
    for pickle_file in os.listdir(dir_path):
        if pickle_file.endswith('.pkl'):
            file_path = os.path.join(dir_path, pickle_file)
            ideal_specs_value, step_len = process_file(file_path)
            if ideal_specs_value is not None:
                if step_len == 1024:
                    fail_specs.append(ideal_specs_value)
                else:
                    achieve_specs.append(ideal_specs_value)

    filtered_fail_specs = [spec for spec in fail_specs if 'Trans_25m_overShoot' not in spec]
    filtered_achieve_specs = [spec for spec in achieve_specs if 'Trans_25m_overShoot' not in spec]
    fail_params_dict = extract_specs(filtered_fail_specs)
    plot_specs(fail_params_dict, 'Failed Specs Distribution')
    achieve_params_dict = extract_specs(filtered_achieve_specs)
    plot_specs(achieve_params_dict, 'Achieved Specs Distribution')
