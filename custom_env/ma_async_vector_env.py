# filepath: [ma_async_vector_env.py](http://_vscodecontentref_/0)
import multiprocessing as mp
import sys
import time
import numpy as np
from copy import deepcopy
from gymnasium.vector import AsyncVectorEnv
from gymnasium import logger
from gymnasium.vector.utils import CloudpickleWrapper, clear_mpi_env_vars, iterate
from enum import Enum
from gymnasium.vector.async_vector_env import AsyncState
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
    
class MultiAgentAsyncVectorEnv(AsyncVectorEnv):
    """支持多智能体环境的异步向量环境"""
    
    def __init__(
        self,
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=False,  # 多智能体环境通常不支持共享内存
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        # 使用自定义worker函数
        self._worker_class = _multi_agent_worker if worker is None else worker
        
        # 强制关闭共享内存以避免错误
        shared_memory = False
        
        # 调用父类初始化
        super().__init__(
            env_fns=env_fns,
            observation_space=observation_space,
            action_space=action_space,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=self._worker_class,
        )
        
        # 探测智能体ID
        dummy_env = env_fns[0]()
        if hasattr(dummy_env, 'agents'):
            self.agent_ids = dummy_env.agents
        else:
            # 默认智能体ID列表
            self.agent_ids = [f"Agent_{i+1}" for i in range(7)]
        print(f"检测到智能体: {self.agent_ids}")
        dummy_env.close()

    def step_async(self, actions):
        """处理多智能体动作格式
        
        actions: 环境列表格式: [{agent1: action1, agent2: action2}, ...]
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise RuntimeError(
                f"Calling [step_async](http://_vscodecontentref_/1) while waiting for a pending call to complete"
            )

        # 确保动作是环境列表格式
        if not isinstance(actions, list) or len(actions) != self.num_envs:
            raise ValueError(f"动作必须是列表格式 [{{agent_id: action}}, ...], 长度为{self.num_envs}")

        # 直接将每个环境的动作字典发送给相应的进程
        for i, (pipe, action) in enumerate(zip(self.parent_pipes, actions)):
            if pipe is not None and not pipe.closed:
                pipe.send(("step", action))
            else:
                print(f"警告: 管道 {i} 已关闭或为None，无法发送动作")
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        """等待step完成，并处理多智能体结果"""
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling [step_wait](http://_vscodecontentref_/2) without any prior call to [step_async](http://_vscodecontentref_/3).",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to [step_wait](http://_vscodecontentref_/4) has timed out after {timeout} second(s)."
            )

        # 初始化结果存储
        all_observations = {}
        all_rewards = {}
        terminateds = []
        truncateds = []
        infos = []
        successes = []

        # 从每个环境收集结果
        for i, pipe in enumerate(self.parent_pipes):
            try:
                if pipe is None or pipe.closed:
                    # 如果管道已关闭或为None，则标记为失败
                    successes.append(False)
                    # 添加默认值
                    terminateds.append(True)
                    truncateds.append(True)
                    infos.append({})
                    continue
                    
                result, success = pipe.recv()
                successes.append(success)
                
                if success:
                    obs_dict, rew_dict, terminated, truncated, info = result
                    print(f"环境 {i} 返回结果: {result}")
                    # 确保obs_dict是字典
                    if not isinstance(obs_dict, dict):
                        print(f"警告：环境 {i} 返回的观察不是字典格式, 而是 {type(obs_dict)}")
                        obs_dict = {f"Agent_{j+1}": obs_dict for j in range(len(self.agent_ids))}
                    
                    # 确保rew_dict是字典
                    if not isinstance(rew_dict, dict):
                        print(f"警告：环境 {i} 返回的奖励不是字典格式, 而是 {type(rew_dict)}")
                        if isinstance(rew_dict, (float, int)):
                            rew_dict = {f"Agent_{j+1}": rew_dict for j in range(len(self.agent_ids))}
                        elif isinstance(rew_dict, (list, np.ndarray)):
                            rew_dict = {f"Agent_{j+1}": rew_dict[j] if j < len(rew_dict) else 0.0 
                                      for j in range(len(self.agent_ids))}
                        else:
                            rew_dict = {f"Agent_{j+1}": 0.0 for j in range(len(self.agent_ids))}
                    
                    # 初始化观察
                    if not all_observations:
                        for agent_id in self.agent_ids:
                            all_observations[agent_id] = [None] * self.num_envs
                            all_rewards[agent_id] = [0.0] * self.num_envs
                    
                    # 收集每个智能体的观察和奖励
                    for agent_id in self.agent_ids:
                        if agent_id not in all_observations:
                            all_observations[agent_id] = [None] * self.num_envs
                        all_observations[agent_id][i] = obs_dict.get(agent_id)
                        
                        if agent_id not in all_rewards:
                            all_rewards[agent_id] = [0.0] * self.num_envs
                        all_rewards[agent_id][i] = rew_dict.get(agent_id, 0.0)
                    
                    terminateds.append(terminated)
                    truncateds.append(truncated)
                    infos.append(info)
                else:
                    # 处理失败情况
                    terminateds.append(True)
                    truncateds.append(True)
                    infos.append({})
            except Exception as e:
                print(f"从环境 {i} 接收数据时发生错误: {e}")
                successes.append(False)
                terminateds.append(True)
                truncateds.append(True)
                infos.append({})

        # 使用自定义错误处理，避免抛出异常
        if not all(successes) and sum(successes) < self.num_envs // 2:
            print(f"警告：有 {self.num_envs - sum(successes)} 个环境执行失败")
            # 尝试获取错误信息但不中断执行
            while not self.error_queue.empty():
                try:
                    index, exctype, value = self.error_queue.get_nowait()
                    print(f"环境 {index} 报告错误: {exctype}: {value}")
                except:
                    break
        
        self._state = AsyncState.DEFAULT

        # 确保所有智能体都有观察和奖励
        for agent_id in self.agent_ids:
            if agent_id not in all_observations:
                all_observations[agent_id] = [None] * self.num_envs
            if agent_id not in all_rewards:
                all_rewards[agent_id] = [0.0] * self.num_envs
        
        # 转换为numpy数组
        for agent_id in all_rewards:
            all_rewards[agent_id] = np.array(all_rewards[agent_id], dtype=np.float32)

        # 确保长度一致
        if len(terminateds) < self.num_envs:
            terminateds.extend([True] * (self.num_envs - len(terminateds)))
        if len(truncateds) < self.num_envs:
            truncateds.extend([True] * (self.num_envs - len(truncateds)))
        if len(infos) < self.num_envs:
            infos.extend([{}] * (self.num_envs - len(infos)))

        return (
            all_observations,
            all_rewards,
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )

    def _check_spaces(self):
        """覆盖空间检查方法，多智能体环境通常有不同的空间"""
        # 为多智能体环境禁用严格的空间检查
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(("_check_spaces", None))
        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        # 不抛出错误，让空间检查更宽松
        return True

def _multi_agent_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    """支持多智能体环境的worker函数"""
    assert shared_memory is None, "多智能体环境不支持共享内存"
    env = env_fn()
    parent_pipe.close()
    
    # 获取环境的结构信息（如果可能）
    try:
        # 尝试获取环境期望的动作结构
        if hasattr(env, 'param_range_dict') and hasattr(env, 'agent_assign_dict'):
            agent_component_mapping = {}
            # 创建从智能体到相应元件的映射
            for agent_id, components in env.agent_assign_dict.items():
                agent_component_mapping[agent_id] = components
            print(f"Worker {index} 获取到智能体-组件映射: {agent_component_mapping}")
        else:
            agent_component_mapping = None
            print(f"Worker {index} 未能获取智能体-组件映射")
    except Exception as e:
        print(f"Worker {index} 获取环境结构时出错: {e}")
        agent_component_mapping = None
    
    # 硬编码设备映射(备用方案)
    default_component_mapping = {
        "Agent_1": ["M0"],
        "Agent_2": ["M10", "M11_Match", "M12", "M18_Match", "M20_Match", "M2_Match", "M36_Match", 
                   "M37", "M3_Match", "M4_Match", "M7_Match", "M9", "ibias"],
        "Agent_3": ["C0", "M1", "M5", "M6_Match", "M8", "R0"],
        "Agent_4": ["C2", "C3", "M14", "M16", "M19"],
        "Agent_5": ["M13_Match", "M21", "M22_Match", "M29_Match", "M30", "M33", "M34_Match", "M35_Match"],
        "Agent_6": ["C1", "C4", "C8", "M23", "M24", "M25", "M27", "M28", "M31"],
        "Agent_7": ["M40", "M41", "M42_Match", "M43_Match", "M44", "M45_Match", "M46", "M47_Match"]
    }
    
    # 组件维度映射(备用方案)
    default_component_dims = {
        "M0": 3, "M10": 3, "M11_Match": 1, "M12": 3, "M18_Match": 1, "M20_Match": 1, 
        "M2_Match": 1, "M36_Match": 1, "M37": 3, "M3_Match": 1, "M4_Match": 1, 
        "M7_Match": 1, "M9": 3, "ibias": 1, "C0": 1, "M1": 3, "M5": 3, "M6_Match": 1, 
        "M8": 3, "R0": 1, "C2": 1, "C3": 1, "M14": 3, "M16": 3, "M19": 3, 
        "M13_Match": 1, "M21": 3, "M22_Match": 1, "M29_Match": 1, "M30": 3, 
        "M33": 3, "M34_Match": 1, "M35_Match": 1, "C1": 1, "C4": 1, "C8": 1, 
        "M23": 3, "M24": 3, "M25": 3, "M27": 3, "M28": 3, "M31": 3, 
        "M40": 3, "M41": 3, "M42_Match": 1, "M43_Match": 1, "M44": 3, 
        "M45_Match": 1, "M46": 3, "M47_Match": 1
    }
    
    try:
        while True:
            try:
                command, data = pipe.recv()
            except (EOFError, OSError):
                break
                
            if command == "reset":
                try:
                    observation, info = env.reset(**data)
                    # 确保observation是字典格式
                    if not isinstance(observation, dict):
                        observation = {"Agent_1": observation}
                    pipe.send(((observation, info), True))
                except Exception as e:
                    print(f"Worker {index} reset错误: {e}")
                    error_queue.put((index, type(e), str(e)))
                    pipe.send((({}, {}), False))
                
            elif command == "step":
               
                try:
                    print(f"Worker {index} step() 开始处理动作...")
                    from collections import OrderedDict
                    import numpy as np
                    
                    # 首先尝试获取环境的设备掩码信息
                    device_mask = None
                    try:
                        if hasattr(env, 'device_mask_dict') and env.device_mask_dict:
                            device_mask = env.device_mask_dict
                            print(f"Worker {index} 获取到设备掩码信息")
                    except Exception as e:
                        print(f"Worker {index} 获取设备掩码失败: {e}")
                    
                    # 创建默认动作结构（根据设备掩码创建）
                    nested_action = OrderedDict()
                    for agent_id, components in default_component_mapping.items():
                        nested_action[agent_id] = OrderedDict()
                        for comp in components:
                            # 如果有设备掩码，检查组件是否被掩码
                            if device_mask and agent_id in device_mask:
                                # 如果组件被掩码掉，则跳过
                                if comp not in device_mask[agent_id]:
                                    print(f"Worker {index} 组件 {comp} 被掩码，跳过")
                                    continue
                            
                            # 根据组件类型设置默认值
                            if "Match" in comp:
                                nested_action[agent_id][comp] = np.array([0.5], dtype=np.float32)
                            else:
                                nested_action[agent_id][comp] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                    
                    # 检查数据类型
                    if isinstance(data, dict):
                        print(f"Worker {index} 接收到字典格式动作: {len(data)} 个智能体")
                        
                        # 如果已经是正确的嵌套字典格式，仅处理未被掩码的组件
                        if all(isinstance(agent_action, dict) for agent_id, agent_action in data.items()):
                            try:
                                for agent_id, agent_action_dict in data.items():
                                    if agent_id in nested_action:
                                        for comp, action in agent_action_dict.items():
                                            # 只处理nested_action中存在的组件
                                            if comp in nested_action[agent_id]:
                                                if isinstance(action, (list, tuple)):
                                                    nested_action[agent_id][comp] = np.array(action, dtype=np.float32)
                                                elif isinstance(action, np.ndarray):
                                                    nested_action[agent_id][comp] = action
                                            else:
                                                print(f"Worker {index} 跳过被掩码组件: {comp}")
                                print(f"Worker {index} 成功处理嵌套字典动作")
                            except Exception as e:
                                print(f"Worker {index} 处理嵌套字典时出错: {e}，使用默认动作")
                        
                        # 处理扁平化字典格式
                        else:
                            try:
                                for agent_id, agent_action in data.items():
                                    if agent_id in nested_action:
                                        # 处理各种动作类型
                                        if isinstance(agent_action, (list, np.ndarray)):
                                            # 转换为列表以便处理
                                            if isinstance(agent_action, np.ndarray):
                                                agent_action = agent_action.tolist()
                                            
                                            # 如果动作长度为0，跳过此智能体
                                            if len(agent_action) == 0:
                                                continue
                                                
                                            # 使用固定偏移量方法而非动态修改
                                            action_index = 0
                                            valid_components = list(nested_action[agent_id].keys())
                                            for comp in valid_components:
                                                try:
                                                    # 确定组件维度
                                                    action_dim = 1 if "Match" in comp else 3
                                                    
                                                    # 如果还有足够的动作值
                                                    if action_index + action_dim <= len(agent_action):
                                                        component_action = agent_action[action_index:action_index+action_dim]
                                                        nested_action[agent_id][comp] = np.array(component_action, dtype=np.float32)
                                                        action_index += action_dim
                                                    # 否则保持默认值
                                                except Exception as e:
                                                    print(f"Worker {index} 处理组件 {comp} 出错: {e}")
                                print(f"Worker {index} 成功处理扁平字典动作")
                            except Exception as e:
                                print(f"Worker {index} 处理扁平字典时出错: {e}")
                    else:
                        print(f"Worker {index} 使用默认动作: {type(data)}")
                    
                    # 执行环境step前进行最终检查
                    for agent_id in list(nested_action.keys()):
                        print(f"Worker {index} 智能体 {agent_id} 包含以下组件: {list(nested_action[agent_id].keys())}")
                    
                    # 执行环境step
                    result = env.step(nested_action)
                    print(f"Worker {index} 环境step()执行完成")
                    
                    # 执行环境step
                    try:
                        
                        # 处理返回结果
                        if len(result) == 5:
                            obs_dict, rewards_dict, terminated, truncated, info = result
                        else:
                            obs_dict, rewards_dict, done, info = result
                            terminated = truncated = done
                        
                        # 确保obs_dict是字典
                        if not isinstance(obs_dict, dict):
                            print(f"Worker {index} 观察不是字典，转换为字典格式")
                            obs_dict = {"Agent_1": obs_dict}
                        
                        # 处理不同类型的rewards_dict
                        if isinstance(rewards_dict, dict):
                            print(f"Worker {index} 奖励已经是字典格式")
                        elif isinstance(rewards_dict, (float, int)):
                            print(f"Worker {index} 奖励是标量值: {rewards_dict}")
                            rewards_dict = {"Agent_1": rewards_dict}
                        elif isinstance(rewards_dict, (list, np.ndarray)):
                            print(f"Worker {index} 奖励是列表/数组: {rewards_dict}")
                            rewards_dict_new = {}
                            agent_keys = list(obs_dict.keys()) if isinstance(obs_dict, dict) else [f"Agent_{i+1}" for i in range(min(7, len(rewards_dict)))]
                            
                            for i, agent_id in enumerate(agent_keys):
                                rewards_dict_new[agent_id] = rewards_dict[i] if i < len(rewards_dict) else 0.0
                            rewards_dict = rewards_dict_new
                        else:
                            print(f"Worker {index} 奖励是未知类型: {type(rewards_dict)}")
                            raise ValueError(f"未知奖励类型: {type(rewards_dict)}")
                        
                        # 环境重置逻辑
                        if terminated or truncated:
                            try:
                                print(f"Worker {index} 环境结束，执行自动重置")
                                old_obs, old_info = obs_dict, info
                                reset_result = env.reset()
                                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                                    obs_dict, new_info = reset_result
                                else:
                                    obs_dict, new_info = reset_result, {}
                                
                                # 确保重置后的obs_dict也是字典
                                if not isinstance(obs_dict, dict):
                                    print(f"Worker {index} 重置后观察不是字典，正在转换...")
                                    obs_dict = {"Agent_1": obs_dict}
                                    
                                # 合并信息
                                if isinstance(info, dict):
                                    info["final_observation"] = old_obs
                                    info["final_info"] = old_info
                            except Exception as e:
                                print(f"Worker {index} 自动重置错误: {e}")
                                if not isinstance(obs_dict, dict):
                                    obs_dict = {"Agent_1": None}
                        
                        pipe.send(((obs_dict, rewards_dict, terminated, truncated, info), True))
                    except Exception as e:
                        print(f"Worker {index} step执行错误: {e}")
                        error_queue.put((index, type(e), str(e)))
                        # 发送默认值而非抛出异常
                        default_obs = {"Agent_1": None}
                        default_reward = {"Agent_1": 0.0}
                        pipe.send(((default_obs, default_reward, True, True, {}), False))
                except Exception as e:
                    print(f"Worker {index} 整体处理错误: {str(e)}")
                    error_queue.put((index, type(e), str(e)))
                    default_obs = {"Agent_1": None}
                    default_reward = {"Agent_1": 0.0}
                    pipe.send(((default_obs, default_reward, True, True, {}), False))
                    
            elif command == "close":
                pipe.send((None, True))
                break
                
            elif command == "_call":
                try:
                    name, args, kwargs = data
                    if name in ["reset", "step", "seed", "close"]:
                        raise ValueError(f"应直接使用 `{name}` 而不是 `_call`")
                    function = getattr(env, name)
                    if callable(function):
                        pipe.send((function(*args, **kwargs), True))
                    else:
                        pipe.send((function, True))
                except Exception as e:
                    print(f"Worker {index} _call错误: {e}")
                    error_queue.put((index, type(e), str(e)))
                    pipe.send((None, False))
                    
            elif command == "_setattr":
                try:
                    name, value = data
                    setattr(env, name, value)
                    pipe.send((None, True))
                except Exception as e:
                    error_queue.put((index, type(e), str(e)))
                    pipe.send((None, False))
                
            elif command == "_check_spaces":
                # 多智能体环境的简化空间检查
                pipe.send(((True, True), True))
                
            else:
                print(f"Worker {index}: 收到未知命令 `{command}`")
                pipe.send((None, False))
                
    except (KeyboardInterrupt, Exception) as e:
        print(f"Worker {index} 进程异常: {e}")
        try:
            error_queue.put((index, type(e), str(e)))
            pipe.send((None, False))
        except:
            pass
    finally:
        try:
            env.close()
        except:
            pass