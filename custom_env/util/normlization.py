def normalization(ideal_specs, cur_specs):
    """
    Normalize the current specs by the ideal specs.
    :param ideal_specs: dict, ideal specs for the circuit
    :param cur_specs: dict, current specs for the circuit
    :return: norm_cur_specs: dict, normalized specs for the circuit
    """

    norm_cur_specs = {}


    for key, sub_dict in cur_specs.items():
        norm_cur_specs_sub = {}
        for sub_key, value in sub_dict.items():
            cur_value = value
            # print(f"Debug, cur_value = {cur_value}")
            ideal_value = ideal_specs[sub_key]['value']
            # print(f"Debug, ideal_value = {ideal_value}")
            norm_value = (cur_value - ideal_value) / (cur_value + ideal_value)
            # print(f"Debug, norm_value = {norm_value}")
            norm_cur_specs_sub[sub_key] = norm_value
        norm_cur_specs[key] = norm_cur_specs_sub

    return norm_cur_specs

# Test Code
# cur_specs = {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': 0, 'gainBandWidth': 0}, 'Trans': {'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081}, 'PSRR': {'powerSupplyRejectionRatio': 9.24983891912481}}
# ideal_specs = {'gainBandWidth': {'constrain_type': 'hard', 'objective': 'max', 'value': 919974068.633527}, 'phaseMargin': {'constrain_type': 'hard', 'objective': 'max', 'value': 74.04052105931964}, 'powerSupplyRejectionRatio': {'constrain_type': 'hard', 'objective': 'max', 'value': 92.7820345842324}, 'pwr': {'constrain_type': 'soft', 'objective': 'min', 'value': 0.0001202291774645289}, 'slewRateDown': {'constrain_type': 'hard', 'objective': 'max', 'value': 6131225.304059564}, 'slewRateUp': {'constrain_type': 'hard', 'objective': 'max', 'value': 4413834.613031412}}
# norm_cur_specs = normalization(ideal_specs, cur_specs)
# print(norm_cur_specs)

# Output
# {'DC': {'pwr': 0.7397158473551917}, 'Stability': {'phaseMargin': -1.0, 'gainBandWidth': -1.0}, 'Trans': {'slewRateUp': 0.061953253129130345, 'slewRateDown': -0.03778817406269507}, 'PSRR': {'powerSupplyRejectionRatio': -0.8186872669975925}}
