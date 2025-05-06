import sys
import os
import time
from collections import deque
from tqdm import tqdm
import heapq
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from shapely.geometry import LinearRing, LineString, Point
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from swanlab.plugin.notification import EmailCallback
import swanlab
from ray import tune
# 导入强化学习相关库
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPOConfig

# 导入自定义环境和工具
from tactics2d.envs import ParkingEnv
from tactics2d.traffic.status import ScenarioStatus
from ParkingWrapper import ParkingWrapper
from RSPlanner import RSPlanner
from Hybrid_policy_agent import RSAgent, ParkingAgent, ParkingActor
from function import execute_path_pid

# 同步SwanLab与TensorBoard
swanlab.sync_tensorboard_torch()

#---------- 配置与初始化部分 ----------#

# 初始化邮件通知插件
email_callback = EmailCallback(
    sender_email="xiaoan_17@qq.com",
    receiver_email="anbc17@seu.edu.cn",
    password="giykurcogkqycbhd",
    smtp_server="smtp.qq.com",
    port=587,
    language="zh",
)

# 初始化SwanLab实验跟踪
swanlab.init(
    project="tactics2d_test",
    experiment_name="test_parking",
    description="基于tactics2d的泊车实验",
    # callbacks=[email_callback],
    mode='disabled'
)

# 环境参数配置
type_proportion = 1.0  # 泊车类型比例，0表示全部为平行泊车，1表示全部为垂直泊车
render_mode = "human"  # 渲染模式，"rgb_array"表示渲染到numpy数组，"human"表示渲染到窗口
render_fps = 50        # 渲染帧率
max_step = 1000        # 每个episode的最大步数
STEER_RATIO = 0.50     # 转向比例因子，实际应用中不使用最大转向角 0.98

#---------- 环境创建与注册部分 ----------#

# 创建泊车环境
env = ParkingEnv(
    type_proportion=type_proportion,
    render_mode=render_mode,
    render_fps=render_fps,
    max_step=max_step,
)

# 获取雷达相关参数
num_lidar_rays = env.scenario_manager._lidar_line  # 360
lidar_obs_shape = num_lidar_rays // 3  # 120
lidar_range = env.scenario_manager._lidar_range

# 打印环境信息
print("1", vars(env))
print("动作空间:", env.action_space)
env = ParkingWrapper(env, lidar_obs_shape)
print("2", vars(env))

# 使用Ray的方式注册环境
def env_creator(env_config):
    """创建并返回包装后的环境实例"""
    new_env = ParkingEnv(
        type_proportion=type_proportion,
        render_mode=render_mode,
        render_fps=render_fps,
        max_step=max_step,
    )
    return ParkingWrapper(new_env, lidar_obs_shape)

# 注册环境到Ray
tune.register_env("CustomParking-v0", env_creator)
print("环境已注册: CustomParking-v0")

# 初始化Reed-Shepp路径规划器
path_planner = RSPlanner(env.unwrapped.scenario_manager.agent, num_lidar_rays, lidar_range, STEER_RATIO)
vehicle = env.unwrapped.scenario_manager.agent

#---------- 训练与评估函数定义 ----------#

def train_rl_agent(env, agent, episode_num=int(1e5), log_path=None, verbose=True):
    """训练强化学习智能体
    
    Args:
        env: 训练环境
        agent: 待训练的智能体
        episode_num: 训练的总episode数
        log_path: 日志保存路径
        verbose: 是否打印详细信息
    
    Returns:
        训练后的智能体
    """
    # 设置日志路径
    if log_path is None:
        log_dir = "./logs"
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_path = os.path.join(log_dir, current_time)
        os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    # 初始化统计数据容器
    reward_list = deque(maxlen=100)
    success_list = deque(maxlen=100)
    loss_list = deque(maxlen=100)
    status_info = deque(maxlen=100)

    step_cnt = 0
    best_success_rate = 0

    print("开始训练!")
    for episode_cnt in tqdm(range(episode_num), desc="评估进度"):
        # 重置环境和智能体
        state, info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        episode_step_cnt = 0

        # 单个episode的训练循环
        while not done:
            step_cnt += 1
            episode_step_cnt += 1
            
            # 转换状态为tensor并确保维度正确
            state_tensor = torch.FloatTensor(state).to(agent.device)
            
            # 选择动作
            action, log_prob, value = agent.choose_action(info, state_tensor)
            
            # 执行动作
            next_state, reward, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            total_reward += reward

            # 存储经验
            observations = [[next_state], [reward], [terminate], [truncated], [info]]
            agent.push([observations, [state], [action], [log_prob], [value]])

            # early stop the episode if the vehicle could not find an available RS path
            if not agent.rs_agent.executing_rs and episode_step_cnt >= 400:
                done = True
                break
            
            if info["scenario_status"] == ScenarioStatus.OUT_BOUND:
                total_reward = -5
                done = True
                break
            
            # 如果收集够了足够的数据，就进行训练
            state = next_state
            loss = agent.train()
            if loss is not None:
                loss_list.append(loss)
        
        # 记录本次episode的结果
        status_info.append([info["scenario_status"], info["traffic_status"]])
        success_list.append(int(info["scenario_status"] == ScenarioStatus.COMPLETED))
        reward_list.append(total_reward)

        # 定期输出训练状态
        if episode_cnt % 10 == 0 and verbose:
            mean_reward = np.mean(reward_list)
            mean_success = np.mean(success_list)
            mean_loss = np.mean(loss_list) if loss_list else 0
            
            print(f"Episode: {episode_cnt}, Total Steps: {step_cnt}")
            print(f"Average Reward: {mean_reward:.2f}")
            print(f"Success Rate: {mean_success:.2%}")
            print(f"Average Loss: {mean_loss:.4f}")
            print("最近10个episode状态:")
            for i in range(min(10, len(status_info))):
                print(f"Reward: {reward_list[-(i+1)]:.2f}, Status: {status_info[-(i+1)]}")
            print("")

            # 记录训练指标到TensorBoard
            writer.add_scalar("average_reward", mean_reward, episode_cnt)
            writer.add_scalar("success_rate", mean_success, episode_cnt)
            writer.add_scalar("average_loss", mean_loss, episode_cnt)
            
            # 保存最佳模型
            if mean_success > best_success_rate:
                best_success_rate = mean_success
                best_model_path = os.path.join(log_path, "best_model.pth")
                agent.save_model(best_model_path)
                print(f"保存最佳模型，成功率: {best_success_rate:.2%}")

    writer.close()
    return agent

def eval_rl_agent(env, agent, episode_num=100, verbose=True):
    """评估智能体的性能
    
    Args:
        env: 评估环境
        agent: 待评估的智能体
        episode_num: 评估的总episode数
        verbose: 是否打印详细信息
    
    Returns:
        成功率和平均奖励
    """
    reward_list = []
    success_list = []
    status_info = []

    print("开始评估!")
    with torch.no_grad():
        for episode in tqdm(range(episode_num), desc="评估进度"):
            # 重置环境和智能体
            state, info = env.reset()
            agent.reset()
            done = False
            total_reward = 0

            # 单个episode的评估循环
            while not done:
                # 转换状态为tensor
                state_tensor = torch.FloatTensor(state).to(agent.device)
                
                # 选择动作
                action, _, _ = agent.choose_action(info, state_tensor)
                
                # 执行动作
                next_state, reward, terminate, truncated, info = env.step(action)
                done = terminate or truncated
                total_reward += reward
                state = next_state

            # 记录结果
            status_info.append([info["scenario_status"], info["traffic_status"]])
            success_list.append(int(info["scenario_status"] == ScenarioStatus.COMPLETED))
            reward_list.append(total_reward)

            # 定期输出评估状态
            if verbose and (episode + 1) % 10 == 0:
                mean_reward = np.mean(reward_list[-10:])
                mean_success = np.mean(success_list[-10:])
                print(f"\nEpisode {episode + 1}")
                print(f"Average Reward (last 10): {mean_reward:.2f}")
                print(f"Success Rate (last 10): {mean_success:.2%}")

    # 计算并输出最终结果
    final_success_rate = np.mean(success_list)
    final_reward = np.mean(reward_list)
    print(f"\n评估完成:")
    print(f"总体成功率: {final_success_rate:.2%}")
    print(f"平均奖励: {final_reward:.2f}")

    return final_success_rate, final_reward

#---------- 智能体配置与训练执行 ----------#

if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

# 智能体配置
agent_config = {
    "debug": True,
    "env": "CustomParking-v0",
    "state_space": env.observation_space,
    "action_space": env.action_space,
    "gamma": 0.995,
    "lr": 1e-4,
    "train_batch_size": 1024,  # 增大到大于默认minibatch_size(128)
    "minibatch_size": 1024,     # 显式设置minibatch_size
    "actor_net": ParkingActor,
    "actor_kwargs": {
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "hidden_size": 256,
    },
    "critic_kwargs": {
        "state_dim": env.observation_space.shape[0], 
        "hidden_size": 256
    },
    "horizon": 20000,
    "batch_size": 1024,
    "adam_epsilon": 1e-8,
}


# 计算车辆参数
min_radius = vehicle.wheel_base / np.tan(vehicle.steer_range[1] * STEER_RATIO)
vehicle_rear_to_center = 0.5 * vehicle.length - vehicle.rear_overhang

# 创建智能体
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rs_agent = RSAgent(path_planner, min_radius, vehicle_rear_to_center, STEER_RATIO)
agent = ParkingAgent(agent_config, rs_agent, device)

# 训练智能体
log_path = "./logs"
num_episode = int(1e5)
train_rl_agent(env, agent, episode_num=num_episode, log_path=log_path, verbose=True)

#---------- 模型评估部分 ----------#

# 加载最佳模型进行评估
start_t = time.time()
try:
    # 加载最佳模型
    model_path = "./logs/best_model.pth"
    agent.load_model(model_path)
    print("开始评估...")
    
    # 执行评估
    succ_rate, avg_reward = eval_rl_agent(env, agent, episode_num=100, verbose=False)
    print("成功率: ", succ_rate)
    print("平均奖励: ", avg_reward)
    print("评估耗时: ", time.time() - start_t)
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("评估耗时: ", time.time() - start_t)