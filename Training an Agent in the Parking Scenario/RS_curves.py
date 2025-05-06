import numpy as np
from tactics2d.envs import ParkingEnv
from ParkingWrapper import ParkingWrapper
from RSPlanner import RSPlanner
import swanlab
from tactics2d.traffic.status import ScenarioStatus
from function import execute_path_pid

# 初始化SwanLab实验跟踪
swanlab.init(
    project="tactics2d_test",
    experiment_name="test_parking_rs_curves",
    description="基于tactics2d的泊车实验",
    # mode='disabled'
)

# 环境参数配置
type_proportion = 1 # 泊车类型比例，0表示全部为平行泊车，1表示全部为垂直泊车
render_mode = "human"  # 渲染模式，"rgb_array"表示渲染到numpy数组，"human"表示渲染到窗口
render_fps = 50        # 渲染帧率
max_step = 1000        # 每个episode的最大步数

# 创建泊车环境
env = ParkingEnv(
    type_proportion=type_proportion,
    render_mode=render_mode,
    render_fps=render_fps,
    max_step=max_step,
)

# 获取雷达相关参数
num_lidar_rays = env.scenario_manager._lidar_line  # 雷达射线数量，360
lidar_obs_shape = num_lidar_rays // 3              # 观测空间中雷达数据的维度，120
lidar_range = env.scenario_manager._lidar_range    # 雷达探测范围

# 转向比例因子，实际应用中不使用最大转向角
STEER_RATIO = 0.5

# 打印环境信息
print("环境初始状态:", vars(env))
print("动作空间:", env.action_space)

# 使用自定义包装器封装环境
env = ParkingWrapper(env, lidar_obs_shape)
print("包装后环境状态:", vars(env))

# 初始化Reed-Shepp路径规划器
path_planner = RSPlanner(env.unwrapped.scenario_manager.agent, num_lidar_rays, lidar_range, STEER_RATIO)
vehicle = env.unwrapped.scenario_manager.agent

# 测试参数设置
num_trails = 100        # 测试次数
num_planning_success = 0  # 规划成功次数计数器
num_parking_success = 0   # 泊车成功次数计数器
max_speed = 0.5        # 泊车任务中的最大速度限制

# 进行多次测试
for n in range(num_trails):
    # 重置环境获取初始状态
    obs, info = env.reset()
    
    # 使用Reed-Shepp算法获取路径规划
    path = path_planner.get_rs_path(info)
    
    if path is None:
        print(f"测试 {n}: 未找到有效路径")
    else:
        # 规划成功，增加计数
        num_planning_success += 1
        print(f"测试 {n}: 找到有效路径")
        
        # 获取起始位置
        start_x, start_y, start_yaw = path_planner.start_pos
        
        # 生成曲线路径点
        path.get_curve_line(np.array([start_x, start_y]), start_yaw, path_planner.radius, 0.1)
        
        # 构建完整轨迹
        traj = [[path.curve[k][0], path.curve[k][1], path.yaw[k]] for k in range(len(path.yaw))]
        print("路径长度: ", path.length)
        
        # 绘制路径并记录到SwanLab
        plot_result = path_planner.plot_rs_path(traj)
        swanlab.log({f"RS曲线路径 {n}": swanlab.Image(plot_result)})

        # 使用PID控制器执行规划的路径
        total_reward, done, info = execute_path_pid(path, env, STEER_RATIO, max_speed)
        
        # 输出场景状态
        print("状态: ", info["scenario_status"])
        
        # 如果任务完成则增加成功计数
        if info["scenario_status"] == ScenarioStatus.COMPLETED:
            num_parking_success += 1

# 输出最终结果统计
print(f"路径规划成功率: {num_planning_success / num_trails:.2f}")
print(f"泊车成功率: {num_parking_success / num_trails:.2f}")
