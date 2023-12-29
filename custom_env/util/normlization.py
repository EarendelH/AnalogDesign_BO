def normalization(ideal_specs, cur_specs):
    """
    Normalize the current specs by the ideal specs.
    :param ideal_specs: dict, ideal specs for the circuit
    :param cur_specs: dict, current specs for the circuit
    :return: norm_cur_specs: dict, normalized specs for the circuit
    """

    norm_cur_specs = {}

    for key1, value1 in cur_specs.items():
        norm_cur_specs[key1] = {}
        for key2, value2 in value1.items():
            ideal_value = ideal_specs[key1][key2]
            norm_value = (value2 - ideal_value) / (value2 + ideal_value)
            norm_cur_specs[key1][key2] = norm_value

    return norm_cur_specs

# Test Code
# cur_specs = {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': 0, 'gainBandWidth': 0},
# 'Trans': {'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081}, 'PSRR': {'powerSupplyRejectionRatio':
# 9.24983891912481}}
# ideal_specs = {'DC': {'pwr': 0.1}, 'Stability': {'phaseMargin': 1, 'gainBandWidth': 1},
# 'Trans': {'slewRateUp': 500.5, 'slewRateDown': 500.5}, 'PSRR': {'powerSupplyRejectionRatio': 10}} print(
# normalization(ideal_specs, cur_specs))

# Output
# {'DC': {'pwr': -0.9840561052972701}, 'Stability': {'phaseMargin': -1.0, 'gainBandWidth': -1.0},
# 'Trans': {'slewRateUp': 0.9997996941628213, 'slewRateDown': 0.9998239295012863}, 'PSRR': {
# 'powerSupplyRejectionRatio': -0.0389697328911101}}
