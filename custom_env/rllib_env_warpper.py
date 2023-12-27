import argparse
import os
import yaml
from collections import OrderedDict
from copy import copy

from ray.tune.registry import get_trainable_cls, register_env

from util.gen_action_sapce import gen_action_space
from util.gen_obs_space import gen_obs_space_extend
from util.gen_param_space import gen_param_space
from util.util_func import create_work_dir
from util.assign_param2netlist import assign_param2netlist
from util.run_spectre_simulation import run_spectre_simulation
from util.cal_reward import cal_reward
from util.generalize_config import generalize_config
from util.update_param import update_parameters

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import air, tune
import ray


class RllibAnalogDesignAutoEnv(MultiAgentEnv):
    def __init__(self, generalize=False, path=''):
        # Init values
        self.cur_param = None
        self.step_num = 0
        self.max_step = 10000

        # Set the ideal specs based on the generalize flag
        self.ideal_specs = generalize_config(generalize, path)

        # Set root directory
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = "run_test"
        self.root_dir_path = os.path.join(file_path, self.root_dir)
        if not os.path.exists(self.root_dir_path):
            raise ValueError(f"Root directory {self.root_dir_path} not found.")

        # Load config files
        self.agent_assign_config = "config/agent_assign.yaml"
        self.result_config = "config/result.yaml"
        self.param_range_config = "config/param_range.yaml"
        self.sim_config = "config/simulation.yaml"

        # Set netlist directory
        self.unassigned_netlist_dir = "netlist_template"

        # Multi-agent config
        with open(self.agent_assign_config, 'r') as file:
            agent_assign = yaml.safe_load(file)

        # RLlib config
        self.possible_agents = list(agent_assign.keys())
        self.agents = self.possible_agents
        self._agent_ids = set(self.agents)

        # Generate param space
        self.param_space = gen_param_space(self.param_range_config)

        # RLlib config
        self.terminateds = set()
        self.truncateds = set()
        self._obs_space_in_preferred_format = True
        self.observation_space = gen_obs_space_extend(self.result_config, self.param_range_config,
                                                      self.agent_assign_config)
        print(f"observation_space: {self.observation_space}")
        self._action_space_in_preferred_format = True
        self.action_space = gen_action_space(self.agent_assign_config)
        print(f"action_space: {self.action_space}")

        super().__init__()

    def reset(self, *, seed=None, options=None):

        # Select the middle point of the param space as the initial param
        init_param = OrderedDict()
        for param, value_list in self.param_space.items():
            n = len(value_list)
            middle_index = n // 2 - 1 if n % 2 == 0 else n // 2
            init_param[param] = value_list[middle_index]
        print(f"Initialing!!!Init param: {init_param}")

        # Generate working directory
        working_dir = create_work_dir(self.root_dir)
        print(f"Initialing!!!Working directory: {working_dir}")

        for unassigned_netlist_file in os.listdir(self.unassigned_netlist_dir):
            if unassigned_netlist_file.endswith(".scs"):
                assigned_netlist_file = unassigned_netlist_file.replace("_parameterized", "")

                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(init_param, unassigned_netlist_file_path, assigned_netlist_file_path,
                                     assigned_yaml_file_path)

        # Run spectre simulation
        with open(self.sim_config, 'r') as file:
            sim_config = yaml.safe_load(file)

        observation = run_spectre_simulation(working_dir, sim_config)

        # Share all observations among agents
        observations = {agent: observation for agent in self.agents}
        print(f"Initialing!!!Simulation result: {observations}")

        self.cur_param = init_param

        return observations, {}

    def step(self, action_dict):

        # Update step number
        self.step_num += 1

        print(f"Updated action: {action_dict} with step number: {self.step_num}")

        # Create working directory
        working_dir = create_work_dir(self.root_dir)
        print(f"Step!!! Working directory: {working_dir} with step number: {self.step_num}")

        print(f"Step!!!Actions: {action_dict} with step number: {self.step_num}")
        # Flatten all actions
        all_action_flatten = OrderedDict()
        for group in action_dict.values():
            for key, value in group.items():
                all_action_flatten[key] = value
        print(f"Step!!!Flatten actions: {all_action_flatten}")

        # Update param with new action
        print(f"Step!!!Current param index: {self.cur_param} with step number: {self.step_num}")
        updated_param = update_parameters(all_action_flatten, self.cur_param, self.param_range_config)
        print(f"Updated param: {updated_param} with step number: {self.step_num}")

        # Parse the updated param and generate the netlist
        for unassigned_netlist_file in os.listdir(self.unassigned_netlist_dir):
            if unassigned_netlist_file.endswith(".scs"):
                assigned_netlist_file = unassigned_netlist_file.replace("_parameterized", "")

                unassigned_netlist_file_path = os.path.join(self.unassigned_netlist_dir, unassigned_netlist_file)
                assigned_netlist_file_path = os.path.join(working_dir, assigned_netlist_file)
                assigned_yaml_file_path = os.path.join(working_dir, "assigned.yaml")
                assign_param2netlist(updated_param, unassigned_netlist_file_path, assigned_netlist_file_path,
                                     assigned_yaml_file_path)

        # Run spectre simulation
        with open(self.sim_config, 'r') as file:
            sim_config = yaml.safe_load(file)

        # Share all observations
        single_obs = run_spectre_simulation(working_dir, sim_config)
        print(f"Step!!!Simulation result: {single_obs} with step number: {self.step_num}")

        obs = {agent: single_obs for agent in self.agents}

        # Store the observation
        with open(os.path.join(working_dir, "result.yaml"), 'w') as file:
            yaml.dump(obs, file)

        # Calculate reward
        rew = {a: -10 for a in self.agents}
        for agent_name in rew:
            rew[agent_name] = cal_reward(self.ideal_specs, obs)

        # Determine termination or truncations
        terminated = {a: False for a in self.agents}
        for agent_name in terminated:
            if rew[agent_name] > 0:
                terminated[agent_name] = True

        truncated = {a: False for a in self.agents}
        for agent_name in truncated:
            if self.step_num >= self.max_step:
                truncated[agent_name] = True

        info = {agent: {} for agent in self.agents}

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        return obs, rew, terminated, truncated, info


def env_creator(env_config):
    return RllibAnalogDesignAutoEnv(generalize=True, path='sampled_specs')


register_env("analog_design_env", env_creator)


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=10)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=10, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=10000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=-0.1,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment("analog_design_env")
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        )
        .training(train_batch_size=1024)
        .rollouts(num_rollout_workers=1, rollout_fragment_length="auto")
        .framework(args.framework)
        .multi_agent(
            policies={"main1", "main2", "main3", "main4"},
            policy_mapping_fn=(lambda aid, episode, worker, **kw: f"main{aid[-1]}"),
            policies_to_train=["main1", "main2", "main3", "main4"],
        )
    )

    results = tune.Tuner(
        args.run,
        run_config=air.RunConfig(
            stop=stop,
        ),
        param_space=config,
    ).fit()

    if not results:
        raise ValueError(
            "No results returned from tune.run(). Something must have gone wrong."
        )
    ray.shutdown()
