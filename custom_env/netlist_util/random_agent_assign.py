import yaml
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_yaml(file_path: str) -> Dict[str, List[str]]:
    """Load YAML file and return the content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, List[str]], file_path: str):
    """Save data to YAML file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def get_all_elements(data: Dict[str, List[str]]) -> List[str]:
    """Extract all elements from all agents."""
    all_elements = []
    for elements in data.values():
        if isinstance(elements, list):  # Ensure we're only adding list items
            all_elements.extend(elements)
    return all_elements


def random_assign(elements: List[str], num_agents: int) -> Dict[str, List[str]]:
    """Randomly assign elements to agents."""
    # Shuffle the elements
    random.shuffle(elements)

    # Create empty lists for each agent
    assignments = {f"Agent_{i + 1}": [] for i in range(num_agents)}

    # Ensure each agent gets at least one element
    for i in range(min(num_agents, len(elements))):
        assignments[f"Agent_{i + 1}"].append(elements[i])

    # Randomly distribute remaining elements
    for element in elements[num_agents:]:
        agent = f"Agent_{random.randint(1, num_agents)}"
        assignments[agent].append(element)

    return assignments


def main(input_path: str):
    # Load the input YAML
    data = load_yaml(input_path)

    # Get the number of agents from the input file
    num_agents = len(data.keys())

    # Get all elements
    all_elements = get_all_elements(data)

    # Perform random assignment
    new_assignments = random_assign(all_elements, num_agents)

    # Generate output path
    input_path = Path(input_path)
    output_path = input_path.parent / f"{input_path.stem}_random{input_path.suffix}"

    # Save the new assignments
    save_yaml(new_assignments, str(output_path))
    print(f"Random assignments saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly reassign elements to agents in a YAML file")
    parser.add_argument("input_path", help="Path to the input YAML file")
    args = parser.parse_args()
    main(args.input_path)