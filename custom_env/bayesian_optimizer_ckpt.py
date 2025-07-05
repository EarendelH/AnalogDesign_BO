from dataclasses import dataclass
import numpy as np
import logging
from typing import List, Optional, Callable, Union, Tuple, Any
import os
from datetime import datetime
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # 用于绘图
import pickle
import json
import swanlab
from tqdm import tqdm

from bo_surrogate_model import SurrogateModel

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    best_params: np.ndarray
    best_value: float
    all_params: List[np.ndarray]
    all_values: List[float]
    optimization_path: List[float]
    epoch_avg_values: List[float]  # 每个epoch的平均reward，而非batch
    all_sim_rewards: List[float]   # 所有仿真的reward值
    sim_indices: List[int]         # 仿真序号索引

class BayesianOptimizer:
    """
    使用神经网络代理模型的贝叶斯优化器

    属性:
        eval_func: 电路评估函数
        parallel_eval_func: 并行批量评估函数
        bounds: 参数边界
        surrogate: 神经网络代理模型
        exploration_weight: 采集函数中的探索权重
        n_restarts: 采集函数优化的随机重启次数
        result_tracker: 结果跟踪和可视化工具
    """

    def __init__(self,
                 eval_func: Callable,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 input_dim: int,
                 save_dir: str,
                 parallel_eval_func: Optional[Callable] = None,
                 exploration_weight: float = 0.1,
                 n_restarts: int = 10,
                 result_tracker: Any = None,
                 use_swanlab: bool = True,
                 swanlab_project: str = "circuit_check_pre",
                 swanlab_run_name: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 resume: bool = False,
                 surrogate_learning_rate: float = 1e-4):
        """
        初始化贝叶斯优化器
        
        Args:
            eval_func: 电路评估函数（单个样本）
            bounds: 参数下限和上限的元组
            input_dim: 输入参数维度
            save_dir: 保存模型和结果的目录
            parallel_eval_func: 并行批量评估函数
            exploration_weight: 探索项的权重
            n_restarts: 采集函数优化的随机重启次数
            result_tracker: 结果跟踪和可视化工具
            use_swanlab: 是否使用swanlab记录实验数据
            swanlab_project: swanlab项目名称
            swanlab_run_name: swanlab运行名称，默认为带时间戳的自动生成名称
            checkpoint_dir: 检查点目录，如果提供则从该目录恢复状态
            resume: 是否从检查点恢复
        """
        self.eval_func = eval_func
        self.parallel_eval_func = parallel_eval_func
        self.bounds = bounds
        self.input_dim = input_dim
        self.save_dir = save_dir
        self.exploration_weight = exploration_weight
        self.n_restarts = n_restarts
        self.result_tracker = result_tracker
        self.resume = resume
        self.checkpoint_dir = checkpoint_dir
        self.surrogate_learning_rate = surrogate_learning_rate

        # 如果resume为True且提供了checkpoint_dir，检查是否存在检查点
        if resume and checkpoint_dir:
            if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint.pkl')):
                self._restore_checkpoint(checkpoint_dir)
                logging.info(f"从检查点恢复状态: {checkpoint_dir}")
                # 使用之前的时间戳
                self.run_timestamp = os.path.basename(checkpoint_dir).replace("bo_run_", "")
                self.save_dir = checkpoint_dir
            else:
                logging.warning(f"未找到检查点文件，创建新的优化会话")
                # 创建带时间戳的保存目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.run_timestamp = timestamp
                self.save_dir = os.path.join(save_dir, f"bo_run_{timestamp}")
                os.makedirs(self.save_dir, exist_ok=True)
        else:
            # 创建带时间戳的保存目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_timestamp = timestamp
            self.save_dir = os.path.join(save_dir, f"bo_run_{timestamp}")
            os.makedirs(self.save_dir, exist_ok=True)

        # 初始化swanlab
        use_swanlab=True
        self.use_swanlab = use_swanlab
        if self.use_swanlab:
            try:
                # import swanlab
                # import swanlab as swanlab
                if self.resume and hasattr(self, 'swanlab_run_id'):
                    # 如果是恢复模式并且有run_id，继续使用相同的run
                    swanlab.init(
                        project=swanlab_project,
                        # id=self.swanlab_run_id,
                        resume="must"
                    )
                    # logging.info(f"swanlab恢复成功，项目: {swanlab_project}, 运行ID: {swanlab.run.id}")
                else:
                    # 否则创建新的run
                    if swanlab_run_name is None:
                        swanlab_run_name = f"bo_run_{self.run_timestamp}"
                    else:
                        swanlab_run_name = f"{swanlab_run_name}_{self.run_timestamp}"
                    
                    swanlab.init(
                        project=swanlab_project,
                        name=swanlab_run_name,
                        config={
                            "input_dim": input_dim,
                            "exploration_weight": exploration_weight,
                            "n_restarts": n_restarts,
                            "bounds": {
                                "lower": bounds[0].tolist() if isinstance(bounds[0], np.ndarray) else bounds[0],
                                "upper": bounds[1].tolist() if isinstance(bounds[1], np.ndarray) else bounds[1],
                            }
                        }
                    )
                    # 保存run_id以便将来恢复
                    # self.swanlab_run_id = swanlab.run.id
                    # logging.info(f"swanlab初始化成功，项目: {swanlab_project}, 运行ID: {swanlab.run.id}")
                    logging.info(f"Swanlab初始化成功，项目{swanlab_project}")
                
                self.swanlab = swanlab
            except ImportError:
                logging.warning("未找到swanlab包，请使用 'pip install swanlab' 安装")
                self.use_swanlab = False
            except Exception as e:
                logging.warning(f"初始化swanlab时出错: {str(e)}")
                self.use_swanlab = False

        # 如果不是恢复模式，初始化优化状态
        if not hasattr(self, 'surrogate'):
            # 初始化代理模型
            
            self.surrogate = SurrogateModel(input_dim=input_dim,learning_rate=self.surrogate_learning_rate)

            # 初始化优化历史
            self.X_history = []
            self.y_history = []
            self.best_value = -np.inf
            self.best_params = None
            
            # 初始化epoch平均reward列表，而非批次平均
            self.epoch_avg_values = []
            self.current_epoch = 0
            
            # 初始化仿真reward跟踪
            self.all_sim_rewards = []  # 所有仿真的reward列表
            self.sim_indices = []      # 仿真计数索引
            
            # 记录开始时间
            self.start_time = time.time()
            # 初始化继续状态
            self.completed_iterations = 0
            self.initial_samples_done = False
            
        logging.info(f"贝叶斯优化器初始化完成，输入维度: {input_dim}")
        
        
    def _save_checkpoint(self):
        """保存优化器状态到检查点"""
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pkl')
        surrogate_path = os.path.join(self.save_dir, 'surrogate_checkpoint.pt')
        
        # 保存代理模型
        self.surrogate.save_model(surrogate_path)
        
        # 保存优化器状态
        checkpoint_data = {
            'X_history': self.X_history,
            'y_history': self.y_history,
            'best_value': self.best_value,
            'best_params': self.best_params,
            'epoch_avg_values': self.epoch_avg_values,
            'current_epoch': self.current_epoch,
            'all_sim_rewards': self.all_sim_rewards,
            'sim_indices': self.sim_indices,
            'completed_iterations': self.completed_iterations,
            'initial_samples_done': self.initial_samples_done,
            'elapsed_time': time.time() - self.start_time
        }
        
        # 保存swanlab运行ID
        if self.use_swanlab and hasattr(self, 'swanlab_run_id'):
            checkpoint_data['swanlab_run_id'] = self.swanlab_run_id
            
        # 写入检查点文件
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        # 写入可读的进度信息
        progress_info = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'completed_iterations': self.completed_iterations,
            'best_value': float(self.best_value),
            'current_epoch': self.current_epoch,
            'elapsed_time_seconds': time.time() - self.start_time,
            'total_simulations': len(self.all_sim_rewards)
        }
        
        with open(os.path.join(self.save_dir, 'progress.json'), 'w') as f:
            json.dump(progress_info, f, indent=2)
            
        logging.info(f"保存检查点到: {checkpoint_path}")
        return checkpoint_path       
    
    
    
    def _restore_checkpoint(self, checkpoint_dir):
        """从检查点恢复优化器状态"""
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            # 恢复基本属性
            self.X_history = checkpoint_data['X_history']
            self.y_history = checkpoint_data['y_history']
            self.best_value = checkpoint_data['best_value']
            self.best_params = checkpoint_data['best_params']
            self.epoch_avg_values = checkpoint_data['epoch_avg_values']
            self.current_epoch = checkpoint_data['current_epoch']
            self.all_sim_rewards = checkpoint_data['all_sim_rewards']
            self.sim_indices = checkpoint_data['sim_indices']
            self.completed_iterations = checkpoint_data['completed_iterations']
            self.initial_samples_done = checkpoint_data['initial_samples_done']
            if 'swanlab_run_id' in checkpoint_data:
                self.swanlab_run_id = checkpoint_data['swanlab_run_id']
            
            # 恢复时间戳
            self.start_time = time.time() - checkpoint_data.get('elapsed_time', 0)
            
            # 初始化代理模型
            self.surrogate = SurrogateModel(input_dim=self.input_dim,learning_rate=self.surrogate_learning_rate)
            
            # 如果存在代理模型检查点，恢复代理模型
            surrogate_path = os.path.join(checkpoint_dir, 'surrogate_checkpoint.pt')
            if os.path.exists(surrogate_path):
                self.surrogate.load_model(surrogate_path)
                
            logging.info(f"从检查点恢复成功，已完成迭代：{self.completed_iterations}，最佳值：{self.best_value:.4f}")
            return True
        except Exception as e:
            logging.error(f"从检查点恢复失败: {str(e)}")
            return False
    
        
    def optimize(self,
                 n_init: int = 10,
                 n_iter: int = 100,
                 batch_size: int = 32,
                 surrogate_epochs: int = 100,
                 save_freq: int = 10) -> OptimizationResult:
        """
        运行贝叶斯优化
        
        Args:
            n_init: 初始随机点数量
            n_iter: 优化迭代次数
            batch_size: 代理模型训练的批量大小
            surrogate_epochs: 代理模型训练的轮数
            save_freq: 保存结果的频率
            
        Returns:
            result: 优化结果
        """
        # 记录swanlab配置
        if self.use_swanlab:
            self.swanlab.config.update({
                "n_init": n_init, 
                "n_iter": n_iter,
                "batch_size": batch_size,
                "surrogate_epochs": surrogate_epochs,
                "save_freq": save_freq
            })
            
        # 根据恢复状态判断是否需要进行初始采样
        if not self.resume or not self.initial_samples_done:
            # 初始随机采样
            logging.info("开始初始随机采样...")
            
            if self.parallel_eval_func:
                # 使用并行评估
                logging.info(f"使用并行评估进行初始采样 (n={n_init})...")
                
                # 生成均匀分布的随机样本
                X_init = np.random.uniform(
                    self.bounds[0],
                    self.bounds[1],
                    size=(n_init, self.input_dim)
                )
                
                # 并行评估样本
                y_init = self.parallel_eval_func(X_init)
                
                # 初始样本视为第0个epoch
                self.current_epoch = 0
                self.epoch_avg_values.append(np.mean(y_init))
                logging.info(f"初始epoch (epoch {self.current_epoch}) 的平均reward: {self.epoch_avg_values[-1]:.4f}")
            else:
                # 顺序评估
                logging.info(f"使用顺序评估进行初始采样 (n={n_init})...")
                X_init = np.random.uniform(
                    self.bounds[0],
                    self.bounds[1],
                    size=(n_init, self.input_dim)
                )
                y_init = np.array([self.eval_func(x) for x in X_init])
                
                # 初始样本视为第0个epoch
                self.current_epoch = 0
                self.epoch_avg_values.append(np.mean(y_init))
                logging.info(f"初始epoch (epoch {self.current_epoch}) 的平均reward: {self.epoch_avg_values[-1]:.4f}")

            # 添加初始rewards到全部仿真历史
            for i, reward in enumerate(y_init):
                self.all_sim_rewards.append(reward)
                self.sim_indices.append(len(self.sim_indices))
                
                # 记录到swanlab
                if self.use_swanlab:
                    self.swanlab.log({
                        "sim_index": i,
                        "reward": float(reward),
                        "phase": "initial_sampling",
                        "best_reward_so_far": float(max(y_init[:i+1]) if i > 0 else reward)
                    })

            # 绘制初始采样后的所有仿真结果
            sim_plot_path = self._plot_all_simulation_rewards()
            logging.info(f"已记录初始{len(y_init)}个仿真点的reward")
            
            # 将图表上传到swanlab
            if self.use_swanlab and sim_plot_path:
                self.swanlab.log({
                    "all_simulations_plot": self.swanlab.Image(sim_plot_path),
                    "epoch": self.current_epoch,
                    "initial_avg_reward": float(self.epoch_avg_values[-1])
                })

            # 更新历史
            self.X_history.extend(X_init)
            self.y_history.extend(y_init)

            # 更新最佳观测
            best_idx = np.argmax(y_init)
            self.best_value = y_init[best_idx]
            self.best_params = X_init[best_idx]
            
            # 绘制初始epoch的平均reward图表
            avg_plot_path = self._plot_epoch_average_rewards()
            
            # 将epoch平均值图表上传到swanlab
            if self.use_swanlab and avg_plot_path:
                self.swanlab.log({
                    "epoch_average_plot": self.swanlab.Image(avg_plot_path),
                    "epoch": self.current_epoch
                })
            
            # 如果有结果跟踪器，初始化它
            if self.result_tracker:
                for i, (params, value) in enumerate(zip(X_init, y_init)):
                    # 将初始样本视为第0次迭代
                    self.result_tracker.update(0, params, value, X_init, y_init)
            
            # 标记初始样本已完成
            self.initial_samples_done = True
            
            # 保存初始采样检查点
            self._save_checkpoint()
            
        # 计算优化路径（恢复的情况下包含历史最佳值）
        optimization_path = [self.best_value]

        # 主优化循环
        logging.info("开始主优化循环...")
        logging.info(f"从迭代 {self.completed_iterations + 1}/{n_iter} 开始...")
        
        for i in range(self.completed_iterations, n_iter):
            # 增加当前epoch计数
            self.current_epoch += 1
            
            # 迭代开始时间
            iter_start_time = time.time()
            logging.info(f"开始第 {i+1}/{n_iter} 次迭代 (epoch {self.current_epoch})...")
        
            try:
                # 训练代理模型
                X_train = np.array(self.X_history)
                y_train = np.array(self.y_history)

                # 创建每次迭代的检查点目录
                model_save_dir = os.path.join(self.save_dir, f"surrogate_iter_{i}")
                
                logging.info(f"训练代理模型，迭代 {i + 1}/{n_iter}... \n当前训练数据形状: {X_train.shape} ")
                
                # 记录训练开始时间
                train_start_time = time.time()
                
                # 训练代理模型并获取训练历史
                history = self.surrogate.fit(
                    X_train,
                    y_train,
                    save_dir=model_save_dir,
                    batch_size=batch_size,
                    epochs=surrogate_epochs,
                    return_history=True  # 要求返回训练历史
                )
                
                # 记录训练结束时间
                train_time = time.time() - train_start_time
                logging.info(f"代理模型训练完成，耗时: {train_time:.2f}秒")
                
                # 如果有训练历史且使用swanlab，记录训练损失
                if self.use_swanlab and history:
                    for epoch, (train_loss, val_loss) in enumerate(zip(history.get('train_loss', []), history.get('val_loss', []))):
                        self.swanlab.log({
                            "epoch": self.current_epoch,
                            "surrogate_epoch": epoch,
                            "surrogate_train_loss": float(train_loss),
                            "surrogate_val_loss": float(val_loss)
                        })

                # 生成多个候选点
                acq_start_time = time.time()
                next_points = self._propose_multiple_points(self.n_restarts)
                acq_time = time.time() - acq_start_time
                logging.info(f"采集函数优化完成，耗时: {acq_time:.2f}秒")
                
                # 记录并行评估开始时间
                eval_start_time = time.time()
                
                # 并行评估候选点
                if self.parallel_eval_func:
                    next_y_values = self.parallel_eval_func(next_points)
                else:
                    # 顺序评估
                    next_y_values = np.array([self.eval_func(x) for x in next_points])
                
                # 记录评估时间
                eval_time = time.time() - eval_start_time
                logging.info(f"候选点评估完成，耗时: {eval_time:.2f}秒")
                
                # 添加新的rewards到全部仿真历史
                for j, reward in enumerate(next_y_values):
                    self.all_sim_rewards.append(reward)
                    sim_idx = len(self.sim_indices)
                    self.sim_indices.append(sim_idx)
                    
                    # 记录到swanlab
                    if self.use_swanlab:
                        self.swanlab.log({
                            "sim_index": sim_idx,
                            "reward": float(reward),
                            "epoch": self.current_epoch,
                            "phase": "optimization",
                            "best_reward_so_far": float(max(self.best_value, max(next_y_values[:j+1])))
                        })

                # 绘制更新后的所有仿真结果
                sim_plot_path = self._plot_all_simulation_rewards()
                logging.info(f"当前已累计记录{len(self.all_sim_rewards)}个仿真点的reward")
                
                # 计算并存储当前epoch的平均reward
                # 这里综合考虑所有历史数据的平均，而不仅仅是当前批次
                all_rewards = np.array(self.y_history + next_y_values.tolist())
                epoch_avg = np.mean(all_rewards)
                self.epoch_avg_values.append(epoch_avg)
                logging.info(f"Epoch {self.current_epoch} 的平均reward: {epoch_avg:.4f}")
                    
                # 找到最佳点
                best_idx = np.argmax(next_y_values)
                next_x = next_points[best_idx]
                next_y = next_y_values[best_idx]

                # 更新历史
                self.X_history.append(next_x)
                self.y_history.append(next_y)

                # 更新最佳观测
                if next_y > self.best_value:
                    improvement = next_y - self.best_value
                    self.best_value = next_y
                    self.best_params = next_x
                    logging.info(f"发现新的最佳值: {self.best_value:.4f} (提升了 {improvement:.4f})")
                    
                    # 记录到swanlab
                    if self.use_swanlab:
                        self.swanlab.log({
                            "epoch": self.current_epoch,
                            "new_best_reward": float(self.best_value),
                            "improvement": float(improvement)
                        })

                # 更新优化路径
                optimization_path.append(self.best_value)
                
                # 更新结果跟踪器
                if self.result_tracker:
                    self.result_tracker.update(i+1, next_x, next_y, next_points, next_y_values)
                
                # 在每次仿真（reward计算后）绘制并保存优化历史图
                plt.figure(figsize=(10,6))
                plt.plot(optimization_path, '-o')
                plt.xlabel('Epoch')
                plt.ylabel('Best Reward')
                plt.title('Optimization Progress')
                plt.grid(True)
                opt_plot_path = os.path.join(self.save_dir, f'optimization_history_epoch_{self.current_epoch}.png')
                plt.savefig(opt_plot_path)
                plt.close()
                logging.info(f"已保存优化历史图: {opt_plot_path}")
                
                # 绘制epoch平均reward图表
                avg_plot_path = self._plot_epoch_average_rewards()
                
                # 记录到swanlab
                if self.use_swanlab:
                    self.swanlab.log({
                        "epoch": self.current_epoch,
                        "best_reward": float(self.best_value),
                        "epoch_avg_reward": float(epoch_avg),
                        "train_time": float(train_time),
                        "acquisition_time": float(acq_time),
                        "evaluation_time": float(eval_time),
                        "iteration_time": float(time.time() - iter_start_time),
                        "total_time": float(time.time() - self.start_time),
                        "optimization_progress_plot": self.swanlab.Image(opt_plot_path),
                        "epoch_average_plot": self.swanlab.Image(avg_plot_path),
                        "all_simulations_plot": self.swanlab.Image(sim_plot_path),
                    })
                
                # 更新完成的迭代次数
                self.completed_iterations = i + 1
                
                # 定期保存检查点
                if (i + 1) % save_freq == 0:
                    self._save_results(i + 1)
                    self._save_checkpoint()
                    
                # 记录迭代时间
                iter_time = time.time() - iter_start_time
                total_time = time.time() - self.start_time
                
                # 详细的性能统计
                logging.info(f"迭代 {i + 1}/{n_iter} (Epoch {self.current_epoch}) 性能统计:")
                logging.info(f"  训练时间: {train_time:.2f}秒 ({train_time/iter_time*100:.1f}%)")
                logging.info(f"  采集函数时间: {acq_time:.2f}秒 ({acq_time/iter_time*100:.1f}%)")
                logging.info(f"  仿真时间: {eval_time:.2f}秒 ({eval_time/iter_time*100:.1f}%)")
                logging.info(f"  总迭代时间: {iter_time:.2f}秒")
                logging.info(f"  累计总时间: {total_time:.2f}秒")
                logging.info(f"  最佳值: {self.best_value:.4f}, 当前epoch平均值: {epoch_avg:.4f}")
            
            except KeyboardInterrupt:
                logging.warning("检测到键盘中断，保存检查点并退出...")
                self._save_checkpoint()
                break
            except Exception as e:
                logging.error(f"迭代 {i+1} 出错: {str(e)}")
                logging.exception("详细异常信息:")
                self._save_checkpoint()
                # 继续下一次迭代
                continue

        # 最终保存
        self._save_results('final')
        self._save_checkpoint()
        
        # 生成最终报告
        if self.result_tracker:
            self.result_tracker.final_report()
            
        # 结束swanlab记录
        if self.use_swanlab:
            # 记录最终结果
            self.swanlab.log({
                "final_best_reward": float(self.best_value),
                "final_epoch_avg_reward": float(self.epoch_avg_values[-1]),
                "total_optimization_time": float(time.time() - self.start_time),
                "total_simulations": len(self.all_sim_rewards)
            })
            
            # 记录最佳参数
            best_params_dict = {}
            for i, param in enumerate(self.best_params):
                best_params_dict[f"best_param_{i}"] = float(param)
            self.swanlab.log(best_params_dict)
            
            # 上传最终CSV文件
            csv_path = os.path.join(self.save_dir, f'all_sim_rewards_final.csv')
            # if os.path.exists(csv_path):
            #     self.swanlab.save(csv_path)
                
            # 完成swanlab运行
            self.swanlab.finish()

        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            all_params=self.X_history,
            all_values=self.y_history,
            optimization_path=optimization_path,
            epoch_avg_values=self.epoch_avg_values,
            all_sim_rewards=self.all_sim_rewards,
            sim_indices=self.sim_indices
        )    
    
    def _plot_all_simulation_rewards(self):
        """
        绘制每个仿真点的reward值
        
        Returns:
            str: 保存的图片路径
        """
        plt.figure(figsize=(10,6))
        
        # 绘制所有仿真点的reward
        plt.scatter(self.sim_indices, self.all_sim_rewards, alpha=0.7, s=15, color='blue', label='reward')
        
        # # 如果点足够多，添加趋势线
        # if len(self.all_sim_rewards) > 5:
        #     z = np.polyfit(self.sim_indices, self.all_sim_rewards, 1)
        #     p = np.poly1d(z)
        #     plt.plot(self.sim_indices, p(self.sim_indices), '-', color='red', 
        #             label=f'趋势线: {z[0]:.4f}x + {z[1]:.4f}')
        
        # 添加最佳值标记
        best_indices = []
        best_rewards = []
        current_best = float('-inf')
        
        for i, reward in enumerate(self.all_sim_rewards):
            if reward > current_best:
                current_best = reward
                best_indices.append(self.sim_indices[i])
                best_rewards.append(reward)
        
        plt.plot(best_indices, best_rewards, 'g-', linewidth=1.5, label='history best reward')
        plt.scatter(best_indices, best_rewards, color='green', s=40, zorder=5)
        
        # plt.xlabel('仿真次数')
        # plt.ylabel('Reward值')
        # plt.title('所有仿真点的Reward值')
        # plt.grid(True)
        # plt.legend()
        
        # # 计算简单统计
        # if len(self.all_sim_rewards) > 0:
        #     mean_reward = np.mean(self.all_sim_rewards)
        #     max_reward = np.max(self.all_sim_rewards)
        #     min_reward = np.min(self.all_sim_rewards)
        #     plt.axhline(y=mean_reward, color='orange', linestyle='--', alpha=0.7)
            
        #     # 添加文本标注
        #     plt.annotate(f'平均值: {mean_reward:.4f}', xy=(0.02, 0.96), xycoords='axes fraction', 
        #                 color='black', backgroundcolor='white', alpha=0.7)
        #     plt.annotate(f'最大值: {max_reward:.4f}', xy=(0.02, 0.92), xycoords='axes fraction',
        #                 color='green', backgroundcolor='white', alpha=0.7)
        #     plt.annotate(f'最小值: {min_reward:.4f}', xy=(0.02, 0.88), xycoords='axes fraction',
        #                 color='red', backgroundcolor='white', alpha=0.7)
        
        # 保存图片
        plot_path = os.path.join(self.save_dir, f'all_simulation_rewards_{len(self.sim_indices)}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"已保存所有仿真reward图: {plot_path}")
        return plot_path
    
    def _plot_epoch_average_rewards(self):
        """
        绘制每个epoch的平均reward
        
        Returns:
            str: 保存的图片路径
        """
        plt.figure(figsize=(10,6))
        
        # 绘制epoch平均reward
        epochs = range(len(self.epoch_avg_values))
        plt.plot(epochs, self.epoch_avg_values, '-o', color='green', label='Epoch Average')
        
        # 如果有3个或以上的点，计算移动平均线
        if len(self.epoch_avg_values) >= 3:
            window_size = min(3, len(self.epoch_avg_values))
            moving_avg = np.convolve(self.epoch_avg_values, np.ones(window_size)/window_size, mode='valid')
            # 绘制移动平均线
            x_vals = range(window_size-1, len(self.epoch_avg_values))
            plt.plot(x_vals, moving_avg, '--', color='red', label=f'{window_size}-point Moving Avg')
        
        plt.xlabel('Epoch Number')
        plt.ylabel('Average Reward')
        plt.title('Epoch Average Rewards')
        plt.grid(True)
        plt.legend()
        
        # 添加数值标签
        for i, avg in enumerate(self.epoch_avg_values):
            plt.annotate(f'{avg:.3f}', (i, avg), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # 保存图片
        plot_path = os.path.join(self.save_dir, f'epoch_average_rewards_{self.current_epoch}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"已保存epoch平均reward图: {plot_path}")
        return plot_path

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        期望改进采集函数
        
        Args:
            x: 要评估的参数
            
        Returns:
            acquisition_value: 采集函数值
        """
        x = x.reshape(1, -1)
        mean, std = self.surrogate.predict(x)

        # 计算改进量
        improvement = mean - self.best_value
        # 添加探索奖励
        exploration_bonus = self.exploration_weight * std

        return -(improvement + exploration_bonus)  # 最小化负期望改进

    def _propose_next_point(self) -> np.ndarray:
        """
        通过优化采集函数提出下一个评估点
        
        Returns:
            next_point: 下一次评估的参数
        """
        best_x = None
        best_acquisition_value = np.inf

        # 多个随机起点
        for _ in range(self.n_restarts):
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])

            # 优化采集函数
            result = minimize(
                self._acquisition_function,
                x0,
                bounds=list(zip(self.bounds[0], self.bounds[1])),
                method='L-BFGS-B'
            )

            if result.fun < best_acquisition_value:
                best_acquisition_value = result.fun
                best_x = result.x

        return best_x
        
    def _propose_multiple_points(self, n_points: int) -> np.ndarray:
        """
        提出多个候选评估点
        
        Args:
            n_points: 要生成的点数量
            
        Returns:
            points: 候选点数组
        """
        points = []
        acquisition_values = []

        # 从多个起点优化采集函数
        for _ in tqdm(range(n_points)):
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])

            # 优化采集函数
            result = minimize(
                self._acquisition_function,
                x0,
                bounds=list(zip(self.bounds[0], self.bounds[1])),
                method='L-BFGS-B'
            )

            points.append(result.x)
            acquisition_values.append(result.fun)

        return np.array(points)

    def _save_results(self, iteration: Union[int, str]):
        """保存优化结果"""
        results = {
            'X_history': self.X_history,
            'y_history': self.y_history,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'epoch_avg_values': self.epoch_avg_values,
            'all_sim_rewards': self.all_sim_rewards,
            'sim_indices': self.sim_indices
        }
        save_path = os.path.join(self.save_dir, f'results_iter_{iteration}.npz')
        np.savez(save_path, **results)
        
        # 同时保存CSV格式便于分析
        csv_path = os.path.join(self.save_dir, f'all_sim_rewards_{iteration}.csv')
        with open(csv_path, 'w') as f:
            f.write('simulation_index,reward\n')
            for idx, reward in zip(self.sim_indices, self.all_sim_rewards):
                f.write(f'{idx},{reward}\n')
        
        logging.info(f"已保存结果: {save_path}")
        logging.info(f"已保存仿真reward CSV: {csv_path}")
        
        # 上传到swanlab
        # if self.use_swanlab:
        #     self.swanlab.save(csv_path)