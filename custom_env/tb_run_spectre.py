from util.run_spectre_simulation import run_dynamic_simulation
import yaml
import os


if __name__ == '__main__':
    sim_config_path = "config_Debashis/simulation.yaml"
    with open(sim_config_path, 'r') as file:
        sim_config = yaml.safe_load(file)
    zero_sim_result = {'DC': {'DC_IQ': 100.0}, 'Stability_1u': {'Stability_1u_phaseMargin': 0.0, 'Stability_1u_gainBandWidth': 0.0}, 'Stability_25m': {'Stability_25m_gainBandWidth': 0.0, 'Stability_25m_phaseMargin': 0.0}, 'Stability_50m': {'Stability_50m_gainBandWidth': 0.0, 'Stability_50m_phaseMargin': 0.0}, 'Load_Reg': {'Load_Reg_loadReg': 100.0}, 'Line_Reg': {'Line_Reg_lineReg': 100.0}, 'Trans_0m': {'Trans_0m_overShoot': 100.0, 'Trans_0m_underShoot': 100.0}, 'Trans_9m': {'Trans_9m_overShoot': 100.0, 'Trans_9m_underShoot': 100.0}, 'Trans_Line_Reg': {'Trans_Line_Reg_overShoot': 100.0, 'Trans_Line_Reg_underShoot': 100.0}, 'PSR': {'PSR_psr_1k': 0.0, 'PSR_psr_1M': 0.0, 'PSR_psr_10M': 0.0}}
    # Iterate over all folders in the select_point folder

    folders = [d for d in os.listdir("/home/wuhan/Downloads/select_point/") if
               os.path.isdir(os.path.join("/home/wuhan/Downloads/select_point/", d))]

    for folder in folders:
        work_dir = os.path.join("/home/wuhan/Downloads/select_point/", folder)
        result, tag = run_dynamic_simulation(work_dir, sim_config, zero_sim_result, show_output=False)
        print(f"Finish {folder} with result \n {result} and tag \n {tag}")