import yaml
import re
import subprocess
import os
import pty
import select
import logging
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Instance:
    name: str
    params: Dict[str, Union[str, float]]

class VirtuosoSession:
    def __init__(self):
        self.process, self.master = self._start_session()

    def _start_session(self) -> Tuple[subprocess.Popen, int]:
        try:
            master, slave = pty.openpty()
            process = subprocess.Popen(['virtuoso', '-nograph'],
                                       stdin=slave, stdout=slave, stderr=slave,
                                       text=True)
            return process, master
        except Exception as e:
            logger.error(f"Error starting Virtuoso session: {e}")
            return None, None

    def wait_for_ready(self) -> None:
        output = ""
        while "> t" not in output:
            rlist, _, _ = select.select([self.master], [], [], 0.1)
            if rlist:
                chunk = os.read(self.master, 1024).decode()
                logger.debug(chunk)
                output += chunk
        os.write(self.master, b'\n')
        os.read(self.master, 1024)  # Read the prompt

    def send_command(self, command: str) -> str:
        os.write(self.master, (command + '\n').encode())
        output = ""
        while not output.strip().endswith(">"):
            rlist, _, _ = select.select([self.master], [], [], 0.1)
            if rlist:
                chunk = os.read(self.master, 1024).decode()
                logger.debug(chunk)
                output += chunk
        return '\n'.join(line for line in output.strip().split('\n') if line.strip() not in (">", command))

    def load_skill_functions(self) -> bool:
        functions = [
            "modifyInstanceParameterWithCallback.il",
            "showCellViewInstances.il",
            "PrintInstanceDetails.il"
        ]
        for func in functions:
            output = self.send_command(f'load("{func}")')
            if "Error" in output:
                logger.error(f"Failed to load {func}")
                return False
        return True

    def close(self):
        if self.process:
            self.process.terminate()
        if self.master:
            os.close(self.master)

    def __enter__(self):
        self.wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ParameterManager:
    UNIT_MAP = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'K': 1e3, 'M': 1e6, 'G': 1e9
    }

    @staticmethod
    def parse_value(value: Union[str, float, int]) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        match = re.match(r"(\d+(?:\.\d+)?)(f|p|n|u|m|k|M|G)?", str(value))
        if match:
            num, unit = match.groups()
            return float(num) * ParameterManager.UNIT_MAP.get(unit, 1)
        return float(value)

    @staticmethod
    def compare_values(yaml_value: Union[str, float], skill_value: str) -> bool:
        yaml_parsed = ParameterManager.parse_value(yaml_value)
        skill_parsed = ParameterManager.parse_value(skill_value)
        return abs(yaml_parsed - skill_parsed) < 1e-15

    @classmethod
    def check_instance_parameters(cls, yaml_data: Dict, skill_output: str, instance_name: str) -> bool:
        yaml_params = yaml_data['Core_Param']
        skill_params = {match.group(1): match.group(2) for match in re.finditer(r'(\w+):\s*"([^"]*)"', skill_output)}

        if instance_name.startswith('M'):
            params_to_check = {'l': 'l', 'w': 'w', 'nf': 'simM' if instance_name == 'MP' else 'fingers'}
            for yaml_key, skill_key in params_to_check.items():
                yaml_value = yaml_params[f'{yaml_key}_{instance_name}{"_per_finger" if yaml_key == "w" else ""}']
                skill_value = skill_params[skill_key]
                if not cls.compare_values(yaml_value, skill_value):
                    logger.error(f"{yaml_key} value mismatch for {instance_name}. YAML: {yaml_value}, Skill: {skill_value}")
                    return False
            if instance_name != 'MP':
                w = cls.parse_value(skill_params['w'])
                fingers = int(skill_params['fingers'])
                calculated_wf = w * fingers
                actual_wf = cls.parse_value(skill_params['wf'])
                if abs(calculated_wf - actual_wf) > 1e-15:
                    logger.error(f"wf value mismatch for {instance_name}. Calculated: {calculated_wf}, Actual: {actual_wf}")
                    return False
        elif instance_name.startswith(('C', 'R')):
            param_key = 'c' if instance_name.startswith('C') else 'r'
            yaml_value = str(yaml_params.get(instance_name, ''))
            skill_value = str(skill_params.get(param_key, ''))
            if not cls.compare_values(yaml_value, skill_value):
                logger.error(f"Value mismatch for {instance_name}. YAML: {yaml_value}, Skill: {skill_value}")
                return False
        return True

def extract_instances(source: Union[str, Dict], is_yaml: bool = True) -> List[Instance]:
    if is_yaml:
        if isinstance(source, str):
            with open(source, 'r') as file:
                data = yaml.safe_load(file)
        else:
            data = source
        core_data = data.get('Core_Param', {})
        instances = []
        for key in core_data.keys():
            match = re.match(r'(w|l|nf)_(M\d+|MP)(?:_per_finger)?', key)
            if match:
                instances.append(Instance(match.group(2), {}))
            elif key.startswith(('C', 'R')):
                instances.append(Instance(key, {}))
        return list({inst.name: inst for inst in instances}.values())
    else:
        return [Instance(line.split(":")[1].strip(), {}) for line in source.split('\n') if line.strip().startswith("Instance:")]

def generate_skill_commands(yaml_data: Dict) -> List[str]:
    commands = []
    lib_name = yaml_data.get('Lib', '')
    core_cell_name = yaml_data.get('Core_Cell', '')
    testbench_cells = yaml_data.get('Testbench_Cell', [])

    for key, value in yaml_data.get('Core_Param', {}).items():
        if key.startswith(('C', 'R')):
            commands.append(f'ModifyInstanceParameter("{lib_name}" "{core_cell_name}" "schematic" "{key}" "{"c" if key.startswith("C") else "r"}" "{value}")')
        else:
            match = re.match(r'(w|l|nf)_(M\d+|MP)(?:_per_finger)?', key)
            if match:
                param_type, instance = match.groups()
                skill_param = {"w": "w", "l": "l", "nf": "simM" if instance == "MP" else "fingers"}[param_type]
                commands.append(f'ModifyInstanceParameter("{lib_name}" "{core_cell_name}" "schematic" "{instance}" "{skill_param}" "{value}")')

    for tb_cell in testbench_cells:
        for instance, value in yaml_data.get('Testbench_Param', {}).items():
            if instance.startswith(('V', 'I')):
                skill_param = "vdc" if instance.startswith('V') else "idc"
                commands.append(f'ModifyInstanceParameter("{lib_name}" "{tb_cell}" "schematic" "{instance}" "{skill_param}" "{value}")')

    return commands

def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "init_param_demo.yaml")

    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    yaml_instances = extract_instances(yaml_data)
    logger.info(f"Instances extracted from YAML: {[inst.name for inst in yaml_instances]}")

    with VirtuosoSession() as session:
        if not session.load_skill_functions():
            logger.error("Failed to load Skill functions. Exiting.")
            return

        lib_name = yaml_data.get('Lib', '')
        core_cell_name = yaml_data.get('Core_Cell', '')

        if not lib_name or not core_cell_name:
            logger.error("Error: Lib or Core_Cell not found in YAML file. Exiting.")
            return

        show_command = f'showCellViewInstances("{lib_name}" "{core_cell_name}" "schematic")'
        output = session.send_command(show_command)

        skill_instances = extract_instances(output, is_yaml=False)
        logger.info(f"Instances extracted from Skill output: {[inst.name for inst in skill_instances]}")

        missing_instances = set(inst.name for inst in yaml_instances) - set(inst.name for inst in skill_instances)
        if missing_instances:
            logger.error(f"The following instances are missing in the schematic: {missing_instances}")
            return

        logger.info("All instances from YAML are present in the schematic. Proceeding with parameter checks.")

        commands = generate_skill_commands(yaml_data)
        for command in commands:
            logger.info(f"Sending command: {command}")
            session.send_command(command)

        for instance in yaml_instances:
            logger.info(f"Checking parameters for instance {instance.name}...")
            command = f'PrintInstanceDetails("{lib_name}" "{core_cell_name}" "schematic" "{instance.name}")'
            output = session.send_command(command)
            if not ParameterManager.check_instance_parameters(yaml_data, output, instance.name):
                logger.error(f"Parameter mismatch for instance {instance.name}.")

        logger.info("All instance parameters match. Parameter modifications complete.")


if __name__ == "__main__":
    main()
