import numpy as np
from PIDController import PIDController
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tactics2d.traffic.status import ScenarioStatus


def rear_center_coord(center_x, center_y, heading, lr):
    """calculate the rear wheel center position based on the center position and heading angle"""
    x = center_x - lr * np.cos(heading)
    y = center_y - lr * np.sin(heading)
    return x, y, heading


def execute_path_pid(path, env, STEER_RATIO, max_speed):
    action_type = {"L": 1, "S": 0, "R": -1}
    radius = env.unwrapped.scenario_manager.agent.wheel_base / np.tan(
        env.unwrapped.scenario_manager.agent.steer_range[1] * STEER_RATIO
    )
    env_agent = env.unwrapped.scenario_manager.agent
    physics_model = env.unwrapped.scenario_manager.agent.physics_model
    # max_speed = max_speed
    max_acceleration = physics_model.accel_range[-1]

    actions = []
    for i in range(len(path.actions)):
        steer_normalized = action_type[path.actions[i]]  # [-1,1]
        distance = path.signs[i] * path.segments[i] * radius
        actions.append([steer_normalized, distance])

    accelerate_controller = PIDController(0, 2.0, 0.0, 0.0)
    velocity_controller = PIDController(0, 0.8, 0.0, 0.0)
    steer_controller = PIDController(0, 5.0, 0.0, 0.0)

    def _calculate_target_point(start_x_, start_y_, start_yaw_, steer, distance):
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

    def _calc_pt_error(pt_start, pt_end, pt, heading):
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

    # calculate target points
    start_state = env_agent.get_state()
    start_x, start_y, start_yaw = start_state.x, start_state.y, start_state.heading
    start_x_, start_y_, start_yaw_ = rear_center_coord(
        start_x, start_y, start_yaw, physics_model.lr
    )
    target_points = []
    arc_centers = []
    for i in range(len(actions)):
        steer, distance = actions[i]
        target_x_, target_y_, target_yaw_, arc_center_x_, arc_center_y_ = _calculate_target_point(
            start_x_, start_y_, start_yaw_, steer, distance
        )
        target_points.append([target_x_, target_y_, target_yaw_])
        arc_centers.append([arc_center_x_, arc_center_y_])
        start_x_, start_y_, start_yaw_ = target_x_, target_y_, target_yaw_

    total_reward = 0
    done = False
    cnt = 0
    for i in range(len(target_points)):
        steer, distance = actions[i]
        base_steer = steer * STEER_RATIO
        vehicle_orientation = np.sign(distance)
        target_x_, target_y_, target_yaw_ = target_points[i]
        arc_center_x_, arc_center_y_ = arc_centers[i]
        curr_state = env_agent.get_state()
        start_x, start_y, start_yaw = curr_state.x, curr_state.y, curr_state.heading
        start_x_, start_y_, start_yaw_ = rear_center_coord(
            start_x, start_y, start_yaw, physics_model.lr
        )
        distance_to_go = np.sqrt((start_x_ - target_x_) ** 2 + (start_y_ - target_y_) ** 2)
        while distance_to_go > 0.02:
            curr_state = env_agent.get_state()
            x, y, yaw, v = curr_state.x, curr_state.y, curr_state.heading, curr_state.speed
            x_, y_, yaw_ = rear_center_coord(x, y, yaw, physics_model.lr)
            distance_to_go = np.sqrt((x_ - target_x_) ** 2 + (y_ - target_y_) ** 2)
            target_v = velocity_controller.update(-distance_to_go * vehicle_orientation, 0)
            target_v = np.clip(target_v, -max_speed, max_speed)
            target_a = accelerate_controller.update(v, target_v)
            target_a = np.clip(target_a, env.action_space.low[1], env.action_space.high[1])

            if arc_center_x_ is not None:
                error_distance_to_center = (
                    np.sqrt((x_ - arc_center_x_) ** 2 + (y_ - arc_center_y_) ** 2) - radius
                )
                error_distance_to_center *= np.sign(
                    steer
                )  # when the error is positive we want to increase the steer
                target_current_yaw = np.arctan2(
                    y_ - arc_center_y_, x_ - arc_center_x_
                ) + np.pi / 2 * np.sign(steer)
            else:
                target_current_yaw = target_yaw_
                error_distance_to_center = _calc_pt_error(
                    [start_x_, start_y_], [target_x_, target_y_], [x_, y_], yaw_
                )

            error_yaw = -(target_current_yaw - yaw_)
            error_yaw = np.arctan2(np.sin(error_yaw), np.cos(error_yaw))
            total_error = error_distance_to_center + 0.5 * error_yaw
            delta_steer = steer_controller.update(-total_error, 0)
            target_steer = base_steer + delta_steer
            # target_steer = np.clip(target_steer, -1, 1)
            target_steer = np.clip(target_steer, env.action_space.low[0], env.action_space.high[0])
            
            action = np.array([target_steer, target_a / max_acceleration], dtype=np.float32)
            _, reward, terminate, truncated, info = env.step(action)
            env.render()
            done = terminate or truncated
            total_reward += reward
            cnt += 1

            if done:
                break

        if done:
            break

    return total_reward, done, info
