# 多智能体GRPO优化(MAGRPO)实现

import time
import logging
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import datetime
from collections import defaultdict
import collections 
from spice_env import RllibAnalogDesignAutoEnv
from ma_async_vector_env import MultiAgentAsyncVectorEnv, _multi_agent_worker

# 尝试导入wandb
try:
    import wandb
    test_run = wandb.init(project="test_init", name="test_init", mode="offline")
    test_run.finish()
    WANDB_AVAILABLE = True
    print("wandb测试初始化成功!")
except:
    WANDB_AVAILABLE = False
    print("wandb未安装或初始化失败，将不会进行实验记录")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置环境变量和日志
os.environ["OPENBLAS_NUM_THREADS"] = "1"
print(f"主进程 PID: {os.getpid()}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 环境配置
ANALOG_CONFIG = {
    "generalize": True,
    "max_step": 1,
    "netlist_folder_name": "netlist_template_Haoqiang",
    "specs_folder_name": "sampled_specs_Haoqiang",
    "config_folder_name": "config_Haoqiang_Regroup", 
    "run_folder_name": f"run_magrpo_Haoqiang",
    "sim_output": False,
    "init_method": "random",
    "corner_sim": False,
    "dc_check": False,
    "region_extract": False,
    "dynamic_queue": False,
    "log_level": "WARNING",
    "reward_func": "cal_reward_Haoqiang",
    "multi_agent_mode": True 
}

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.state_dim = state_dim
        
        # 更宽的层
        self.fc1 = torch.nn.Linear(state_dim, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        
        # 使用LayerNorm来稳定训练
        self.layer_norm1 = torch.nn.LayerNorm(512)
        self.layer_norm2 = torch.nn.LayerNorm(512)
        self.layer_norm3 = torch.nn.LayerNorm(256)
        
        self.fc_mu = torch.nn.Linear(256, action_dim)
        self.fc_std = torch.nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = F.relu(self.layer_norm3(self.fc3(x)))
        
        mu = torch.sigmoid(self.fc_mu(x))
        std_raw = self.fc_std(x)
        # 扩大标准差范围到[0.1, 0.6]
        std = 0.1 + 0.5 * torch.sigmoid(std_raw)
        
        return mu, std

def make_env(config, env_id=0):
    """创建单个环境的工厂函数"""
    def _init():
        # 对每个实例使用唯一文件夹
        env_config = config.copy()
        env_config["run_folder_name"] = f"{config['run_folder_name']}"
        env = RllibAnalogDesignAutoEnv(env_config)
        return env
    return _init

def create_vector_analog_env(num_envs=8, config=None):
    """创建向量化环境"""
    if config is None:
        config = ANALOG_CONFIG.copy()
        config["multi_agent_mode"] = True  # 确保环境支持多智能体
    
    # 创建环境工厂函数列表
    env_fns = [make_env(config, i) for i in range(num_envs)]
    
    # 使用自定义的多智能体向量环境
    vector_env = MultiAgentAsyncVectorEnv(
        env_fns,
        shared_memory=False, 
        copy=True,
        daemon=True,
        context="spawn"  # 使用spawn而非fork以提高稳定性
    )
    logger.info(f"创建了包含 {num_envs} 个并行环境的MultiAgentAsyncVectorEnv")
    return vector_env

def flatten_dict_observation(observation, agent_id=None):
    """将字典或元组观察展平为向量"""
    # 处理None情况
    if observation is None:
        print(f"警告：观察值为None，使用默认空数组")
        return np.array([0.0], dtype=np.float32)
    
    # 处理OrderedDict类型
    if isinstance(observation, collections.OrderedDict):
        print(f"处理OrderedDict观察，键: {list(observation.keys())}")
        values = []
        for k, v in sorted(observation.items()):
            if isinstance(v, np.ndarray):
                values.extend(v.flatten())
            else:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
        return np.array(values, dtype=np.float32)
    
    # 处理gymnasium.spaces.Dict类型 - 这是空间定义而非观察值
    if isinstance(observation, gym.spaces.Dict):
        print(f"警告：接收到空间对象而非观察值：{type(observation)}")
        return np.array([0.0], dtype=np.float32)
    
    # 处理元组类型观察
    if isinstance(observation, tuple):
        print(f"处理元组形式观察，长度: {len(observation)}")
        values = []
        for arr in observation:
            if isinstance(arr, np.ndarray):
                values.extend(arr.flatten())
            else:
                try:
                    values.append(float(arr))
                except (ValueError, TypeError):
                    continue
        return np.array(values, dtype=np.float32)

        

def get_agent_action_dims(env):
    """获取每个智能体的动作维度"""
    agent_action_dims = {}
    
    print(f"环境类型: {type(env)}")
    print(f"环境action_space: {env.action_space}")
    
    # RllibAnalogDesignAutoEnv环境可能有不同的接口
    if hasattr(env, 'agents'):
        print(f"环境agents属性: {env.agents}")
        # 使用env.agents作为智能体ID列表
        for agent_id in env.agents:
            if hasattr(env, 'agent_action_dims') and agent_id in env.agent_action_dims:
                # 如果环境直接提供动作维度
                agent_action_dims[agent_id] = env.agent_action_dims[agent_id]
    
    # 如果上面的方法没有检测到智能体，使用我们已知的结构
    if not agent_action_dims:
        # 手动设置已知的智能体动作空间
        agent_action_dims = {
            "Agent_1": 3,
            "Agent_2": 21,
            "Agent_3": 12,
            "Agent_4": 11,
            "Agent_5": 14,
            "Agent_6": 21,
            "Agent_7": 16
        }
        print(f"使用手动指定的智能体动作维度: {agent_action_dims}")
    
    return agent_action_dims

def save_checkpoint(models, optimizers, training_state, path):
    """保存训练检查点"""
    checkpoint = {
        'models_state_dict': {agent_id: model.state_dict() for agent_id, model in models.items()},
        'optimizers_state_dict': {agent_id: opt.state_dict() for agent_id, opt in optimizers.items()},
        'training_state': training_state,
        'random_states': {
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    }
    torch.save(checkpoint, path)
    logger.info(f"保存检查点到: {path}")

def load_checkpoint(path, models, optimizers):
    """加载训练检查点"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点文件 {path} 不存在")
    
    logger.info(f"从检查点恢复: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    for agent_id, model in models.items():
        if agent_id in checkpoint['models_state_dict']:
            model.load_state_dict(checkpoint['models_state_dict'][agent_id])
    
    for agent_id, optimizer in optimizers.items():
        if agent_id in checkpoint['optimizers_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizers_state_dict'][agent_id])
    
    # 恢复随机数生成器状态
    if 'random_states' in checkpoint:
        np.random.set_state(checkpoint['random_states']['numpy'])
        torch.set_rng_state(checkpoint['random_states']['torch'])
        if torch.cuda.is_available() and checkpoint['random_states']['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['random_states']['torch_cuda'])
    
    return checkpoint['training_state']

def collect_multi_agent_trajectories(envs, policies, agent_action_dims, num_steps=10, gamma=0.99):
    """从向量化环境采样多个智能体的轨迹"""
    group_size = envs.num_envs
    print(f"并行环境数量: {group_size}")
    seed_num = np.random.randint(0, 1000)
    
    # 重置环境获取初始观察
    print("重置环境...")
    obs_dict, _ = envs.reset(seed=[seed_num] * group_size)
    print(f"初始观察类型: {type(obs_dict)}")
    print(f"初始观察结构: {obs_dict.keys() if isinstance(obs_dict, dict) else '非字典类型'}")
    
    if isinstance(obs_dict, dict):
        print(f"观察键: {list(obs_dict.keys())}")
        # 打印第一个智能体的观察结构
        if "Agent_1" in obs_dict:
            agent_obs = obs_dict["Agent_1"]
            print(f"Agent_1观察类型: {type(agent_obs)}")
            if isinstance(agent_obs, list) and len(agent_obs) > 0:
                print(f"Agent_1第一个环境观察: {type(agent_obs[0])}")
                # 如果是字典，打印其键
                if isinstance(agent_obs[0], dict):
                    print(f"Agent_1观察键: {list(agent_obs[0].keys())}")
                    # 打印第一个键的值
                    first_key = list(agent_obs[0].keys())[0]
                    print(f"键'{first_key}'的值: {agent_obs[0][first_key]}")
    
    # 创建每个智能体的轨迹存储
    all_agent_trajectories = {}
    all_agent_rewards = {}
    simulation_steps = 0
    
    # 确保初始化所有智能体的轨迹字典
    for agent_id in policies.keys():
        all_agent_trajectories[agent_id] = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": torch.zeros(group_size, device=device)
        }
    
    # 初始化观察状态 - 处理不同观察格式
    agent_states = {}
    for agent_id, policy in policies.items():
        # 从策略网络获取正确的输入维度
        expected_dim = policy.state_dim
        
        # 为每个智能体创建正确维度的空观察
        agent_states[agent_id] = np.zeros((group_size, expected_dim), dtype=np.float32)
        
        # 尝试填充有效的观察值
        if isinstance(obs_dict, dict):
            if agent_id in obs_dict:
                # 直接使用智能体特定观察
                agent_obs = obs_dict[agent_id]
                for env_idx in range(group_size):
                    if isinstance(agent_obs, list) and len(agent_obs) > env_idx:
                        flat_obs = flatten_dict_observation(agent_obs[env_idx])
                        # 安全地复制可用的观察值
                        copy_dims = min(len(flat_obs), expected_dim)
                        agent_states[agent_id][env_idx, :copy_dims] = flat_obs[:copy_dims]
            else:
                # 使用共享观察
                for env_idx in range(group_size):
                    env_obs = {}
                    for feature_key, feature_values in obs_dict.items():
                        if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > env_idx:
                            env_obs[feature_key] = feature_values[env_idx]
                    
                    flat_obs = flatten_dict_observation(env_obs)
                    # 安全地复制可用的观察值
                    copy_dims = min(len(flat_obs), expected_dim)
                    agent_states[agent_id][env_idx, :copy_dims] = flat_obs[:copy_dims]
    
    # 打印观察维度信息
    for agent_id in policies.keys():
        if agent_id in agent_states:
            print(f"智能体 {agent_id} 的states shape: {agent_states[agent_id].shape}")
   
    for t in range(num_steps):
        # 临时存储每个智能体在所有环境的动作
        agent_actions = {}
        
        # 为每个智能体计算动作
        for agent_id, policy in policies.items():
            states_tensor = torch.tensor(agent_states[agent_id], dtype=torch.float32, device=device)
            mu, sigma = policy(states_tensor)
            action_dist = torch.distributions.Normal(mu, sigma)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions).detach()
            
            # 保存该智能体的轨迹数据
            all_agent_trajectories[agent_id]["states"].append(agent_states[agent_id])
            all_agent_trajectories[agent_id]["actions"].append(actions)
            all_agent_trajectories[agent_id]["log_probs"].append(log_probs)
            
            # 将动作转为CPU NumPy数组
            agent_actions[agent_id] = actions.cpu().numpy()
        
        # *** 关键修改: 正确格式化动作为AsyncVectorEnv需要的格式 ***
        # 对于每个并行环境，创建一个包含所有智能体动作的字典
        env_actions_list = []
        for env_idx in range(group_size):
            env_action = {}
            for agent_id in policies.keys():
                env_action[agent_id] = agent_actions[agent_id][env_idx].tolist()  # 确保是列表而非数组
            env_actions_list.append(env_action)
        
        print(f"动作格式示例 (环境0): {env_actions_list[0]}")
        
        # 直接使用正确格式的动作调用step
        next_obs_dict, rewards, terminations, truncations, infos = envs.step(env_actions_list)
        
        # 统计模拟步数
        simulation_steps += group_size - np.count_nonzero(terminations)
        
        for agent_id in policies.keys():
            if agent_id in rewards:
                # 确保奖励张量与环境数量一致
                agent_rewards = rewards[agent_id]
                if isinstance(agent_rewards, np.ndarray) and len(agent_rewards) == group_size:
                    agent_rewards = torch.tensor(agent_rewards, device=device)
                else:
                    # 如果长度不匹配，创建正确尺寸的奖励张量
                    agent_rewards = torch.zeros(group_size, device=device)
                    if isinstance(agent_rewards, np.ndarray):
                        # 填充可用的值
                        valid_len = min(len(agent_rewards), group_size)
                        agent_rewards[:valid_len] = torch.tensor(
                            agent_rewards[:valid_len], device=device)
                
                # 安全添加奖励 - 确保维度相同
                if agent_rewards.shape[0] == all_agent_trajectories[agent_id]["rewards"].shape[0]:
                    all_agent_trajectories[agent_id]["rewards"] += agent_rewards
                else:
                    print(f"警告: 智能体 {agent_id} 的奖励尺寸不匹配 ({agent_rewards.shape[0]} vs {all_agent_trajectories[agent_id]['rewards'].shape[0]})")
                    # 使用正确大小的零奖励来保持训练稳定
                    all_agent_trajectories[agent_id]["rewards"] += torch.zeros_like(
                        all_agent_trajectories[agent_id]["rewards"])
            else:
                print(f"警告: 智能体 {agent_id} 不在奖励字典中")
        
        # 如果所有环境都终止，则提前结束
        if np.all(terminations):
            simulation_steps += group_size
            break
    
    # 完成轨迹收集，规范化数据格式
    for agent_id in policies.keys():
        # 归一化奖励
        normalized_rewards = (all_agent_trajectories[agent_id]["rewards"] / num_steps)
        
        # 转换轨迹数据为正确的张量格式
        all_states = torch.tensor(np.array(all_agent_trajectories[agent_id]["states"]), device=device).permute(1, 0, 2)
        all_log_probs = torch.stack(all_agent_trajectories[agent_id]["log_probs"]).permute(1, 0, 2)
        all_actions = torch.stack(all_agent_trajectories[agent_id]["actions"]).permute(1, 0, 2)
        
        # 更新轨迹字典
        all_agent_trajectories[agent_id] = {
            "all_states": all_states,
            "all_log_probs": all_log_probs,
            "all_actions": all_actions,
            "normalized_rewards": normalized_rewards
        }
        
        # 保存总奖励用于评估
        all_agent_rewards[agent_id] = (normalized_rewards * num_steps).cpu().numpy()
    
    # 添加仿真步数
    all_agent_trajectories["simulation_steps"] = simulation_steps
    
    return all_agent_trajectories, all_agent_rewards

def calc_advantages_with_gapo(trajectories, agent_id):
    """从轨迹中提取该智能体的奖励，并标准化"""
    rewards = trajectories[agent_id]["normalized_rewards"]
    mean_reward = torch.mean(rewards)
    std_reward = torch.std(rewards) + 1e-8
    advantages = (rewards - mean_reward) / std_reward
    return advantages

def gapo_update_multi_agent(all_trajectories, policies, optimizers, n_iterations=10, eps=0.2):
    """
    使用GAPO更新多个智能体的策略
    """
    agent_losses = {}
    
    # 为每个智能体分别更新策略
    for agent_id, policy in policies.items():
        if agent_id not in all_trajectories:
            logger.warning(f"跳过智能体 {agent_id} 的更新 - 轨迹数据缺失")
            continue
            
        trajectories = all_trajectories[agent_id]
        
        # 计算标准化的优势
        advantages = calc_advantages_with_gapo(all_trajectories, agent_id).unsqueeze(-1)
        
        # 提取轨迹数据
        all_states = trajectories["all_states"]
        all_log_probs = trajectories["all_log_probs"]
        all_chosen_actions = trajectories["all_actions"]
        batch_size = len(all_states)
        
        optimizer = optimizers[agent_id]
        agent_loss = 0
        
        # 进行多次迭代更新
        for i_iter in range(n_iterations):
            loss = 0
            for i in range(len(all_states)):
                states = all_states[i]
                log_probs = all_log_probs[i]
                chosen_actions = all_chosen_actions[i]
                advantage = advantages[i]
                
                # 计算新的动作概率
                mu, sigma = policy(states)
                action_dist = torch.distributions.Normal(mu, sigma)
                new_log_probs = action_dist.log_prob(chosen_actions)
                
                # 计算比率和裁剪目标
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
                trajectory_loss = torch.mean(-torch.min(surr1, surr2))
                loss += trajectory_loss
            
            # 归一化损失
            loss /= batch_size
            
            # 更新策略
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            agent_loss = loss.item()
        
        agent_losses[agent_id] = agent_loss
    
    return agent_losses

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="多智能体GAPO训练模拟电路设计环境")
    parser.add_argument("--group_size", type=int, default=32, help="并行环境数量")
    parser.add_argument("--episodes", type=int, default=300, help="训练回合数")
    parser.add_argument("--max_steps", type=int, default=1, help="每个环境的最大步数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--save_freq", type=int, default=10, help="保存模型频率")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    
    args = parser.parse_args()
    
    # 创建向量化环境
    group_size = args.group_size
    envs = create_vector_analog_env(num_envs=group_size)
    
    # 创建示例环境以获取观察和动作空间维度
    sample_env = RllibAnalogDesignAutoEnv(ANALOG_CONFIG)
    sample_obs = sample_env.observation_space.sample()
    agent_action_dims = get_agent_action_dims(sample_env)
    
    # 获取智能体ID列表
    agent_ids = list(agent_action_dims.keys())
    if not agent_ids:
        raise ValueError("无法检测到任何智能体!")
    
    logger.info(f"检测到智能体: {agent_ids}")
    logger.info(f"每个智能体的动作维度: {agent_action_dims}")
    DEFAULT_OBS_DIM = 195 
    print(f"样本观察类型: {type(sample_obs)}")
    if isinstance(sample_obs, collections.OrderedDict):
        print(f"样本观察键: {list(sample_obs.keys())}")
    else:
        print(f"警告: 样本观察不是OrderedDict类型")

    agent_obs_dims = {}
    for agent_id in agent_ids:
        agent_obs_dims[agent_id] = DEFAULT_OBS_DIM
        print(f"智能体 {agent_id} 使用默认观察维度: {DEFAULT_OBS_DIM}")
    logger.info(f"每个智能体的观察维度: {agent_obs_dims}")
    sample_env.close()
    
    policies = {}
    optimizers = {}
    
    for agent_id in agent_ids:
        obs_dim = agent_obs_dims[agent_id]
        act_dim = agent_action_dims[agent_id]
        policies[agent_id] = PolicyNetContinuous(obs_dim, act_dim).to(device)
        optimizers[agent_id] = torch.optim.Adam(policies[agent_id].parameters(), lr=args.lr)
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./magrpo_weights/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = f"MAGRPO_analog_{timestamp}"
        wandb.init(
            project="multi-agent-analog-circuit-design",
            name=run_name,
            config={
                "agent_ids": agent_ids,
                "group_size": group_size,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "learning_rate": args.lr,
                "agent_obs_dims": agent_obs_dims,
                "agent_action_dims": agent_action_dims
            }
        )
        logger.info("Wandb记录已启用")
    
    # 从检查点恢复（如果指定）
    start_episode = 0
    return_history = {agent_id: [] for agent_id in agent_ids}
    best_rewards = {agent_id: float('-inf') for agent_id in agent_ids}
    team_reward_history = []  # 添加团队奖励历史记录
    best_team_reward = float('-inf')  # 初始化团队最佳奖励
    
    if args.resume and args.checkpoint:
        try:
            training_state = load_checkpoint(args.checkpoint, policies, optimizers)
            start_episode = training_state.get('episode', 0)
            return_history = training_state.get('return_history', return_history)
            best_rewards = training_state.get('best_rewards', best_rewards)
            # 恢复团队指标
            team_reward_history = training_state.get('team_reward_history', [])
            best_team_reward = training_state.get('team_best_reward', float('-inf'))
            logger.info(f"从回合 {start_episode} 恢复训练")
        except Exception as e:
            logger.error(f"恢复训练失败: {e}")
            logger.info("从头开始训练")
    
    # 训练参数
    max_steps = args.max_steps
    episode_num = args.episodes
    
    # 开始训练
    start = time.time()
    total_simulations = 0
    
    # 训练循环
    for i_episode in tqdm(range(start_episode, episode_num)):
        # 使用GAPO收集多智能体轨迹
        all_trajectories, all_rewards = collect_multi_agent_trajectories(
            envs, policies, agent_action_dims, max_steps
        )
        
        episode_simulations = all_trajectories.get("simulation_steps", max_steps * group_size)
        total_simulations += episode_simulations
        
        # 使用GAPO更新多个智能体的策略
        agent_losses = gapo_update_multi_agent(all_trajectories, policies, optimizers)
        all_agent_rewards_list = []
        all_agent_losses_list = []
      

        # 更新统计信息
        for agent_id in agent_ids:
            if agent_id in all_rewards:
                avg_reward = np.mean(all_rewards[agent_id])
                return_history[agent_id].append(avg_reward)
                all_agent_rewards_list.append(avg_reward)
                if agent_id in agent_losses:
                    all_agent_losses_list.append(agent_losses[agent_id])
                
                # 记录到wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        f"agent_{agent_id}/avg_reward": avg_reward,
                        f"agent_{agent_id}/min_reward": np.min(all_rewards[agent_id]),
                        f"agent_{agent_id}/max_reward": np.max(all_rewards[agent_id]),
                        f"agent_{agent_id}/loss": agent_losses.get(agent_id, 0),
                        "episode": i_episode,
                        "total_simulations": total_simulations
                    })
                    
        if all_agent_rewards_list:
            team_avg_reward = np.mean(all_agent_rewards_list)
            team_min_reward = np.min(all_agent_rewards_list)
            team_max_reward = np.max(all_agent_rewards_list)
            team_reward_history.append(team_avg_reward)  # 添加到历史记录
            # 更新团队最佳奖励
            if team_avg_reward > best_team_reward:
                best_team_reward = team_avg_reward
                # 保存团队最佳模型
                for agent_id in agent_ids:
                    best_model_path = f"{save_dir}/magrpo_agent_{agent_id}_team_best.pth"
                    torch.save(policies[agent_id].state_dict(), best_model_path)
                logger.info(f"新的团队最佳模型已保存，奖励: {team_avg_reward:.4f}")
            
            
            # 记录总体指标到wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "team/avg_reward": team_avg_reward,
                    "team/min_reward": team_min_reward,
                    "team/max_reward": team_max_reward,
                    "team/best_reward": best_team_reward,
                    "team/avg_loss": np.mean(all_agent_losses_list) if all_agent_losses_list else 0,
                    "training/episode_simulations": episode_simulations,
                    "training/total_simulations": total_simulations,
                    "episode": i_episode,  # 添加episode作为明确的指标
                }, step=i_episode)  # 使用相同的step值
            logger.info(f'团队平均奖励: {team_avg_reward:.4f}, 历史最佳: {best_team_reward:.4f}')
                # # 保存最佳模型
                # if avg_reward > best_rewards[agent_id]:
                #     best_rewards[agent_id] = avg_reward
                #     best_model_path = f"{save_dir}/magrpo_agent_{agent_id}_best.pth"
                #     torch.save(policies[agent_id].state_dict(), best_model_path)
                #     logger.info(f"智能体 {agent_id} 新的最佳模型已保存: {best_model_path}")
        
        # 定期保存模型和检查点
        if (i_episode + 1) % args.save_freq == 0:
            # 保存模型
            for agent_id in agent_ids:
                model_path = f"{save_dir}/magrpo_agent_{agent_id}_episode_{i_episode+1}.pth"
                torch.save(policies[agent_id].state_dict(), model_path)
            
            # 保存检查点
            checkpoint_path = f"{save_dir}/checkpoint_episode_{i_episode+1}.pt"
            training_state = {
                'episode': i_episode + 1,
                'return_history': return_history,
                'best_rewards': best_rewards,
                'team_best_reward': best_team_reward,  # 添加团队最佳奖励
                'team_reward_history': team_reward_history  # 添加团队奖励历史
            }
            save_checkpoint(policies, optimizers, training_state, checkpoint_path)
            logger.info(f"检查点已保存到: {checkpoint_path}")
        
        # 打印进度
        agent_rewards_str = ", ".join([f"{agent_id}: {return_history[agent_id][-1]:.4f}" for agent_id in agent_ids])
        logger.info(f'第 {i_episode} 次试验, 奖励: {agent_rewards_str}')
    
    logger.info(f"训练完成，用时: {time.time() - start:.2f}秒")
    
    # 保存最终模型
    for agent_id in agent_ids:
        final_model_path = f"{save_dir}/magrpo_agent_{agent_id}_final.pth"
        torch.save(policies[agent_id].state_dict(), final_model_path)
    
    logger.info(f"所有最终模型已保存到: {save_dir}")
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 8))
    for agent_id in agent_ids:
        if return_history[agent_id]:
            plt.plot(range(len(return_history[agent_id])), return_history[agent_id], label=f"Agent {agent_id}")
    
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('MAGRPO on Analog Circuit')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/magrpo_learning_curve.png")
    
    # 关闭wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # 关闭环境
    envs.close()

if __name__ == "__main__":
    main()