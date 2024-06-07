def norm_ideal_spec(input_specs, norm_specs):
    """
    Normalize the current specs by the ideal specs.
    :param input_specs: dict, ideal specs for the circuit
    :param norm_specs: dict, current specs for the circuit
    :return: norm_cur_specs: dict, normalized specs for the circuit
    """

    norm_cur_specs = {}

    for key, sub_dict in norm_specs.items():
        norm_cur_specs_sub = {}
        for sub_key, value in sub_dict.items():
            ideal_value = value
            print(f"Debug, cur_value = {ideal_value}")
            input_value = input_specs[sub_key]['value']
            print(f"Debug, ideal_value = {input_value}")
            norm_value = (input_value - ideal_value) / (input_value + ideal_value)
            print(f"Debug, norm_value = {norm_value}")
            norm_cur_specs_sub[sub_key] = norm_value
        norm_cur_specs[key] = norm_cur_specs_sub

    return norm_cur_specs


def norm_sim_spec(sim_specs, norm_specs):
    norm_cur_specs = {
        outer_k: {
            inner_k: (sim_specs[outer_k][inner_k] - norm_specs[outer_k][inner_k]) / (
                        sim_specs[outer_k][inner_k] + norm_specs[outer_k][inner_k])
            for inner_k in sim_specs[outer_k]
        }
        for outer_k in sim_specs
    }
    return norm_cur_specs

# Test Code

# input_specs = {'gainBandWidth': {'constrain_type': 'hard', 'objective': 'max', 'value': 919974068.633527},
# 'phaseMargin': {'constrain_type': 'hard', 'objective': 'max', 'value': 74.04052105931964},
# 'powerSupplyRejectionRatio': {'constrain_type': 'hard', 'objective': 'max', 'value': 92.7820345842324},
# 'pwr': {'constrain_type': 'soft', 'objective': 'min', 'value': 0.0001202291774645289}, 'slewRateDown': {
# 'constrain_type': 'hard', 'objective': 'max', 'value': 6131225.304059564}, 'slewRateUp': {'constrain_type': 'hard',
# 'objective': 'max', 'value': 4413834.613031412}}

# norm_specs = {'DC': {'pwr': 0.005}, 'PSRR': {'powerSupplyRejectionRatio': 45}, 'Stability': {'gainBandWidth':
# 500000000.0, 'phaseMargin': 45.0}, 'Trans': {'slewRateDown': 5000000.0, 'slewRateUp': 5000000.0}}

# norm_cur_specs = normalization(input_specs, norm_specs)
# print(norm_cur_specs)

# Output

# {'DC': {'pwr': 0.7397158473551917}, 'Stability': {'phaseMargin': -1.0, 'gainBandWidth': -1.0}, 'Trans': {
# 'slewRateUp': 0.061953253129130345, 'slewRateDown': -0.03778817406269507}, 'PSRR': {'powerSupplyRejectionRatio':
# -0.8186872669975925}}
