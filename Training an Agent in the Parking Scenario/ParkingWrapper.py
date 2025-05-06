import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


# the wrapper is used to preprocess the observation and action
class ParkingWrapper(Wrapper):
    def __init__(self, env: gym.Env, lidar_obs_shape):
        super().__init__(env)
        self.lidar_obs_shape = lidar_obs_shape
        observation_shape = (lidar_obs_shape + 6,)  # 120: lidar obs size. 6: additional features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )

    def _preprocess_action(self, action):
        """将动作从[-1,1]映射到实际的动作空间"""
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1, 1)
        action_space = self.env.action_space
        action = (
            action * (action_space.high - action_space.low) / 2
            + (action_space.high + action_space.low) / 2
        )
        return action

    def _preprocess_observation(self, obs, info):
        """预处理观察数据，统一维度"""
        # 处理激光雷达数据
        lidar_info = np.clip(info["lidar"], 0, 20)
        lidar_info = lidar_info[::3]  # 从360降采样到120
        lidar_feature = lidar_info / 20.0  # 归一化到[0, 1]

        # 处理其他特征
        other_feature = np.array([
            info["diff_position"] / 10.0,  # 归一化距离
            np.cos(info["diff_angle"]),
            np.sin(info["diff_angle"]),
            np.cos(info["diff_heading"]),
            np.sin(info["diff_heading"]),
            info["state"].speed,
        ], dtype=np.float32)

        # 合并特征并确保维度正确
        observation = np.concatenate([lidar_feature, other_feature])
        return observation.astype(np.float32)

    def reset(self, **kwargs):
        """重置环境并返回初始观察"""
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._preprocess_observation(obs, info)
        return processed_obs, info

    def step(self, action):
        """执行一步动作并返回结果"""
        processed_action = self._preprocess_action(action)
        next_obs, reward, terminated, truncated, info = self.env.step(processed_action)
        processed_obs = self._preprocess_observation(next_obs, info)
        return processed_obs, reward, terminated, truncated, info

