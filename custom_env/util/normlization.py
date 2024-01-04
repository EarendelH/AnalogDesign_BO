def normalization(ideal_specs, cur_specs):
    """
    Normalize the current specs by the ideal specs.
    :param ideal_specs: dict, ideal specs for the circuit
    :param cur_specs: dict, current specs for the circuit
    :return: norm_cur_specs: dict, normalized specs for the circuit
    """

    norm_cur_specs = {
        outer_k: {
            inner_k: (cur_specs[outer_k][inner_k] - ideal_specs[outer_k][inner_k]) / (
                        cur_specs[outer_k][inner_k] + ideal_specs[outer_k][inner_k])
            for inner_k in cur_specs[outer_k]
        }
        for outer_k in cur_specs
    }

    return norm_cur_specs

# Test Code
# cur_specs = {'DC': {'pwr': 0.000803601}, 'Stability': {'phaseMargin': 0, 'gainBandWidth': 0},
# 'Trans': {'slewRateUp': 4996857.610474619, 'slewRateDown': 5684722.22222081}, 'PSRR': {'powerSupplyRejectionRatio':
# 9.24983891912481}}
# ideal_specs = {'DC': {'pwr': 0.005}, 'PSRR': {'powerSupplyRejectionRatio': 45}, 'Stability': {
# 'gainBandWidth': 500000000.0, 'phaseMargin': 45.0}, 'Trans': {'slewRateDown': 5000000.0, 'slewRateUp': 5000000.0}}
# norm_cur_specs = normalization(ideal_specs, cur_specs)
# print(norm_cur_specs)

# Output
# {'DC': {'pwr': -0.723068143382014}, 'Stability': {'phaseMargin': -1.0, 'gainBandWidth': -1.0},
# 'Trans': {'slewRateUp': -0.00031433772969701487, 'slewRateDown': 0.0640842324189586}, 'PSRR': {
# 'powerSupplyRejectionRatio': -0.6589911010458707}}
