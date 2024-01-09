import datetime
import os
import random


def unit_conversion(value):
    """
    Convert unit-suffixed values to float.
    Handles units from micro (u) to giga (G).
    """

    if isinstance(value, int):
        return float(value)

    unit_mapping = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, "M": 1e6, 'G': 1e9
    }
    for unit, multiplier in unit_mapping.items():
        if isinstance(value, str) and value.endswith(unit):
            return float(value.replace(unit, '')) * multiplier

    return float(value)


# Test Code
# test_values = ['0.5u', '10M', '2k', 5, '7']
# converted_values = [unit_conversion(value) for value in test_values]
# print(converted_values)

# Output
# [5e-07, 10000000.0, 1000000000.0, 100.0, 2000.0]

def create_work_dir(base_path):
    """
    Create a new directory for the work.
    :param base_path: base path of the work directory
    :return: work_dir: path of the work directory
    """
    cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    random_num = str(random.randint(1000, 9999))
    dir_name = f"tmp_{cur_time}{random_num}"
    work_dir = os.path.join(base_path, dir_name)
    os.makedirs(work_dir, exist_ok=True)
    # print(f"Created working directory: {work_dir}")
    return work_dir
