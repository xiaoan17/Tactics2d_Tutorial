import torch
import numpy as np
import os
from torch.distributions import Normal
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPO
from PIDController import PIDController


class RSAgent:
    def __init__(
        self,
        path_planner,
        execute_radius,
        dr,
        steer_ratio,
        max_acceleration=2.0,
        max_speed=0.5,
    ):
        self.path_planner = path_planner
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.steer_ratio = steer_ratio
        self.execute_radius = execute_radius
        self.dr = dr  # the distance between the rear wheel center and the vehicle center
        self.distance_record = []

        self.accelerate_controller = PIDController(0, 2.0, 0.0, 0.0)
        self.velocity_controller = PIDController(0, 0.8, 0.0, 0.0)
        self.steer_controller = PIDController(0, 5.0, 0.0, 0.0)

        self.path_info = {
            "segments": [],
            "target_points": [],
            "arc_centers": [],
            "start_points": [],
        }

    @property
    def executing_rs(self):
        return len(self.path_info["segments"]) > 0

    def _rear_center_coord(self, center_x, center_y, heading, lr):
        """calculate the rear wheel center position based on the center position and heading angle"""
        x = center_x - lr * np.cos(heading)
        y = center_y - lr * np.sin(heading)
        return x, y, heading

    def _calc_pt_error(self, pt_start, pt_end, pt, heading):
        # calculate the distance from pt to the line defined by pt_start and pt_end
        x1, y1 = pt_start
        x2, y2 = pt_end
        x0, y0 = pt
        yaw = np.arctan2(y2 - y1, x2 - x1)
        orientation = 1 if np.cos(yaw - heading) > 0 else -1
        # if heading is roughly from pt_start to pt_end, then the error should be positive when pt is on the left side of the line
        error_distance_to_line = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1
        error_distance_to_line /= np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        error_distance_to_line *= orientation
        return error_distance_to_line

    def _calculate_target_point(self, start_x_, start_y_, start_yaw_, steer, distance):
        radius = self.execute_radius
        if steer == 0:
            target_x_ = start_x_ + distance * np.cos(start_yaw_)
            target_y_ = start_y_ + distance * np.sin(start_yaw_)
            target_yaw_ = start_yaw_
            arc_center_x_, arc_center_y_ = None, None
        elif steer == 1:
            arc_center_x_ = start_x_ - radius * np.sin(start_yaw_)
            arc_center_y_ = start_y_ + radius * np.cos(start_yaw_)
            delta_radian = distance / radius
            target_x_ = arc_center_x_ + radius * np.sin(start_yaw_ + delta_radian)
            target_y_ = arc_center_y_ - radius * np.cos(start_yaw_ + delta_radian)
            target_yaw_ = start_yaw_ + delta_radian
        elif steer == -1:
            arc_center_x_ = start_x_ + radius * np.sin(start_yaw_)
            arc_center_y_ = start_y_ - radius * np.cos(start_yaw_)
            delta_radian = distance / radius
            target_x_ = arc_center_x_ + radius * np.sin(-start_yaw_ + delta_radian)
            target_y_ = arc_center_y_ + radius * np.cos(-start_yaw_ + delta_radian)
            target_yaw_ = start_yaw_ - delta_radian

        return target_x_, target_y_, target_yaw_, arc_center_x_, arc_center_y_

    def calculate_target_points(self, path, start_pos):
        """
        Calculate the end points of each segment in the path.
        """
        action_type = {"L": 1, "S": 0, "R": -1}
        radius = self.execute_radius

        segments = []
        for i in range(len(path.actions)):
            # print(path.actions[i], path.signs[i], path.segments[i], path.segments[i]* radius)
            steer_normalized = action_type[path.actions[i]]  # [-1,1]
            correction_speed_ratio_on_curve = 1 if steer_normalized == 0 else 1.0  # / np.cos(beta)
            distance = path.signs[i] * path.segments[i] * radius * correction_speed_ratio_on_curve
            segments.append([steer_normalized, distance])

        start_x, start_y, start_yaw = start_pos
        start_x_, start_y_, start_yaw_ = self._rear_center_coord(start_x, start_y, start_yaw, self.dr)
        target_points = []
        arc_centers = []
        start_points = []
        for i in range(len(segments)):
            steer, distance = segments[i]
            target_x_, target_y_, target_yaw_, arc_center_x_, arc_center_y_ = (
                self._calculate_target_point(start_x_, start_y_, start_yaw_, steer, distance)
            )
            target_points.append([target_x_, target_y_, target_yaw_])
            arc_centers.append([arc_center_x_, arc_center_y_])
            start_points.append([start_x_, start_y_, start_yaw_])
            start_x_, start_y_, start_yaw_ = target_x_, target_y_, target_yaw_

        self.path_info = {
            "segments": segments,
            "target_points": target_points,
            "arc_centers": arc_centers,
            "start_points": start_points,
        }

    def get_action(self, curr_state):
        """
        In this function, we calculate the action based on the current state of the vehicle and the control information calculated by the path planner
        """
        if not self.executing_rs:
            return None

        curr_x, curr_y, curr_yaw, curr_v = (
            curr_state.x,
            curr_state.y,
            curr_state.heading,
            curr_state.speed,
        )
        curr_x_, curr_y_, curr_yaw_ = self._rear_center_coord(curr_x, curr_y, curr_yaw, self.dr)
        distance_to_go = np.sqrt(
            (curr_x_ - self.path_info["target_points"][0][0]) ** 2
            + (curr_y_ - self.path_info["target_points"][0][1]) ** 2
        )
        last_distance = self.distance_record[-1] if len(self.distance_record) > 0 else np.inf
        self.distance_record.append(distance_to_go)
        if distance_to_go < 0.02 or (last_distance < distance_to_go and distance_to_go < 0.1):
            self.distance_record = []
            self.path_info["segments"].pop(0)
            self.path_info["target_points"].pop(0)
            self.path_info["arc_centers"].pop(0)
            self.path_info["start_points"].pop(0)

        if not self.executing_rs:
            return np.array([0, 0], dtype=np.float32)

        steer, distance = self.path_info["segments"][0]
        base_steer = steer * self.steer_ratio
        radius = self.execute_radius
        vehicle_orientation = np.sign(distance)
        target_x_, target_y_, target_yaw_ = self.path_info["target_points"][0]
        arc_center_x_, arc_center_y_ = self.path_info["arc_centers"][0]

        distance_to_go = np.sqrt((curr_x_ - target_x_) ** 2 + (curr_y_ - target_y_) ** 2)
        target_v = self.velocity_controller.update(-distance_to_go * vehicle_orientation, 0)
        target_v = np.clip(target_v, -self.max_speed, self.max_speed)
        target_a = self.accelerate_controller.update(curr_v, target_v)
        target_a = np.clip(target_a, -self.max_acceleration, self.max_acceleration)

        if arc_center_x_ is not None:
            error_distance_to_center = (
                np.sqrt((curr_x_ - arc_center_x_) ** 2 + (curr_y_ - arc_center_y_) ** 2) - radius
            )
            error_distance_to_center *= np.sign(
                steer
            )  # when the error is positive we want to increase the steer
            target_current_yaw = np.arctan2(
                curr_y_ - arc_center_y_, curr_x_ - arc_center_x_
            ) + np.pi / 2 * np.sign(steer)
        else:
            target_current_yaw = target_yaw_
            start_x_, start_y_, _ = self.path_info["start_points"][0]
            error_distance_to_center = self._calc_pt_error(
                [start_x_, start_y_], [target_x_, target_y_], [curr_x_, curr_y_], curr_yaw_
            )

        error_yaw = -(target_current_yaw - curr_yaw_)
        error_yaw = np.arctan2(np.sin(error_yaw), np.cos(error_yaw))
        total_error = error_distance_to_center + 0.5 * error_yaw
        delta_steer = self.steer_controller.update(-total_error, 0)
        target_steer = base_steer + delta_steer
        target_steer = np.clip(target_steer, -0.52, 0.52)


        action = np.array([target_steer, target_a / self.max_acceleration], dtype=np.float32)

        return action

    def plan(self, info):
        path = self.path_planner.get_rs_path(info)
        if path is None:
            return False
        start_state = info["state"]
        start_pos = [start_state.x, start_state.y, start_state.heading]
        self.calculate_target_points(path, start_pos)
        return True

    def reset(self):
        self.accelerate_controller.reset()
        self.velocity_controller.reset()
        self.steer_controller.reset()
        for key in self.path_info.keys():
            self.path_info[key] = []
        self.distance_record = []


class ParkingActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ParkingActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络层
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # 均值和标准差输出层
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """前向传播，确保输入维度正确"""
        # 确保输入是2D张量 [batch_size, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        features = self.net(state)
        mean = torch.tanh(self.mean_layer(features))  # 使用tanh限制在[-1, 1]范围
        return mean
    
    def get_dist(self, state):
        """获取动作分布"""
        mean = self.forward(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)
    
    def action(self, state):
        """采样动作并计算对数概率"""
        dist = self.get_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # 合并多维动作的概率
        return action, log_prob

    def evaluate(self, states, actions):
        """评估动作的对数概率和熵"""
        dist = self.get_dist(states)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_probs, entropy

# 实现Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 处理输入形状
        if x.dim() == 2 and x.shape[1] == 1:
            x = x.squeeze(1)  # 将(126,1)转换为(126,)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 将(126,)转换为(1,126)

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        value = self.value(x)
        return value

class ParkingAgent(PPO):
    def __init__(self, config, rs_agent: RSAgent, device, max_speed=0.5, max_acceleration=2.0):
        super(ParkingAgent, self).__init__(config)
        self.rs_agent = rs_agent
        self.accel_controller = PIDController(0, 2.0, 0.0, 0.0)
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.action_cnt = 0
        self.last_action = None
        self.device = device
        self.config = config

        # 初始化轨迹缓冲区
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        # PPO参数
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("lambda", 0.95)
        self.batch_size = config.get("train_batch_size", 512)
        
        # 添加这些代码来初始化actor_net和critic_net
        # 从配置中获取网络参数
        actor_class = config.get("actor_net", ParkingActor)
        actor_kwargs = config.get("actor_kwargs", {})
        critic_kwargs = config.get("critic_kwargs", {})
        
        # 初始化actor网络
        self.actor_net = actor_class(**actor_kwargs).to(device)
        
        # 初始化critic网络
        self.critic_net = Critic(**critic_kwargs).to(device)
        
        # 设置优化器
        lr = config.get("lr", 5e-5)
        adam_epsilon = config.get("adam_epsilon", 1e-8)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), lr=lr, eps=adam_epsilon
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=lr, eps=adam_epsilon
        )

    def control_rlagent_action(self, info, action):
        """
        The network is trained to output the accel and steer ratio, we need to limit the speed to interact with the environment.
        """
        action_shape = action.shape
        if len(action_shape) > 1:
            assert action_shape[0] == 1
            action = action.squeeze(0)
        curr_v = info["state"].speed
        max_positive_v = self.max_speed
        max_negative_v = -self.max_speed
        max_accel = self.accel_controller.update(curr_v, max_positive_v)
        max_decel = self.accel_controller.update(curr_v, max_negative_v)
        target_a = np.clip(action[1] * self.max_acceleration, max_decel, max_accel)
        action[1] = target_a / self.max_acceleration

        return action.reshape(*action_shape)

    def RL_get_action(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        action, log_prob = self.actor_net.action(states)
        value = self.critic_net(states)
    
        if isinstance(action, torch.Tensor):
            # 先转到CPU，再转为numpy
            action = action.detach().cpu().numpy().astype(np.float32)
    
        # 确保动作范围和类型正确
        action = np.clip(action, [-0.52, -2], [0.52, 2]).astype(np.float32)
        
        # 转换其他返回值
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.detach().cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().flatten()

        return action, log_prob, value
    
    def choose_action(self, info, state):
        """
        选择执行动作的策略：
        1. 如果RS路径可用（正在执行或可以规划），则使用RS代理的输出
        2. 否则使用RL代理的输出
        
        参数:
            info: 环境信息字典，包含当前状态和场景信息
            state: 当前状态向量
            
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        # 检查是否正在执行RS路径或者能否规划新的RS路径
        if self.rs_agent.executing_rs or self.rs_agent.plan(info):
            # 使用RS代理生成动作
            action = self.rs_agent.get_action(info["state"])
            # 评估该动作的对数概率和价值
            # print('action', action)
            log_prob, value = self.evaluate_action(state, action)
        else:
            # 使用RL策略网络生成动作
            action, log_prob, value = self.RL_get_action(state)
            # 注释掉的代码用于控制RL代理动作的输出范围
            # action = self.control_rlagent_action(info, action)
    
        action = np.clip(action[0], [-0.52, -2], [0.52, 2]).astype(np.float32)
        return action, log_prob, value

    def evaluate_action(self, states, actions: np.ndarray):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        log_prob, _ = self.actor_net.evaluate(states, actions)
        value = self.critic_net(states)

        log_prob = log_prob.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()

        return log_prob, value

    def reset(self):
        self.accel_controller.reset()
        self.rs_agent.reset()
    
    def save_model(self, path):
        """保存模型参数和配置
        
        Args:
            path: 保存路径，例如 './checkpoints/model.pth'
        """
        # 确保路径存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 收集需要保存的状态
        checkpoint = {
            'actor_state_dict': self.actor_net.state_dict(),
            'critic_state_dict': self.critic_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'device': str(self.device),
            'max_speed': self.max_speed,
            'max_acceleration': self.max_acceleration
        }
        
        # 保存模型
        try:
            torch.save(checkpoint, path)
            print(f"模型已成功保存到: {path}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
    
    def load_model(self, path):
        """加载模型参数和配置
        
        Args:
            path: 模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        try:
            # 加载checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # 加载网络参数
            self.actor_net.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_net.load_state_dict(checkpoint['critic_state_dict'])
            
            # 加载优化器状态
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # 加载其他配置
            self.max_speed = checkpoint['max_speed']
            self.max_acceleration = checkpoint['max_acceleration']
            
            print(f"模型已成功从 {path} 加载")
        except Exception as e:
            print(f"加载模型时出错: {e}")
    

    def push(self, transition):
        """存储一个时间步的数据
        
        Args:
            transition: [observations, states, actions, log_probs, values]
                observations: [next_states, rewards, terminates, truncated, infos]
        """
        observations, states, actions, log_probs, values = transition
        next_state, reward, terminate, truncated, _ = observations
        
        # 存储到轨迹缓冲区
        self.trajectory_buffer['states'].append(states[0])  # 取第一个元素因为是列表
        self.trajectory_buffer['actions'].append(actions[0])
        self.trajectory_buffer['rewards'].append(reward[0])
        self.trajectory_buffer['next_states'].append(next_state[0])
        self.trajectory_buffer['dones'].append(terminate[0] or truncated[0])
        self.trajectory_buffer['values'].append(values[0])

        # 在push方法中
        if isinstance(log_probs[0], (list, np.ndarray)) and len(log_probs[0]) > 0 and isinstance(log_probs[0][0], (list, np.ndarray)):
            # 如果是二维的情况
            self.trajectory_buffer['log_probs'].append(log_probs[0][0])
        else:
            # 如果是一维的情况
            self.trajectory_buffer['log_probs'].append(log_probs[0])

    
    def train(self):
        """执行PPO训练"""
        # 检查是否收集了足够的数据
        if len(self.trajectory_buffer['states']) < self.batch_size:
            return None
            
        # 计算优势估计和回报
        advantages, returns = self._compute_advantages_and_returns()

        # 准备训练数据，注意处理数据格式
        states = torch.FloatTensor(np.array(self.trajectory_buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.trajectory_buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.trajectory_buffer['log_probs'])).to(self.device)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # PPO更新
        total_loss = self._update_policy(states, actions, old_log_probs, advantages, returns)
        
        # 清空轨迹缓冲区
        # self._clear_buffer()
        
        return total_loss.item()
    
    def _compute_advantages_and_returns(self):
        """计算GAE优势估计和回报"""
        rewards = np.array(self.trajectory_buffer['rewards'])
        values = np.array(self.trajectory_buffer['values'])
        dones = np.array(self.trajectory_buffer['dones'])
        
        # 计算GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards)-1)):
            next_value = values[t+1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        # 计算回报
        returns = advantages + values
        
        return advantages, returns
    
    def _clear_buffer(self):
        """清空轨迹缓冲区"""
        for key in self.trajectory_buffer:
            self.trajectory_buffer[key] = []
    
    def _update_policy(self, states, actions, old_log_probs, advantages, returns):
        """执行PPO策略更新"""
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次更新
        for _ in range(self.config.get("num_sgd_iter", 10)):
            # 计算新的动作概率和值
            new_log_probs, entropy = self.actor_net.evaluate(states, actions)
            values = self.critic_net(states)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO裁剪目标
            clip_param = self.config.get("clip_param", 0.2)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 值函数损失
            critic_loss = 0.5 * (returns - values).pow(2).mean()
            
            # 熵奖励
            entropy_loss = -self.config.get("entropy_coeff", 0.01) * entropy.mean()
            
            # 总损失
            total_loss = actor_loss + critic_loss + entropy_loss
            
            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        return total_loss