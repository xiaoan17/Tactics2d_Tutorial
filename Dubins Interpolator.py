import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import Polygon
import matplotlib.lines as mlines

from tactics2d.math.interpolate import *

def get_bbox(center, length, width, heading):
    """创建车辆边界框多边形
    
    Args:
        center: 车辆中心点坐标
        length: 车辆长度
        width: 车辆宽度
        heading: 车辆朝向角度(弧度)
    
    Returns:
        车辆边界框的坐标列表
    """
    # 创建矩形多边形并根据中心点进行平移
    polygon = Polygon(
        np.array(
            [
                (length / 2, width / 2),
                (-length / 2, width / 2),
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
            ]
        )
        + center
    )
    # 根据朝向角度旋转多边形
    polygon = rotate(polygon, heading, origin="center", use_radians=True)

    return list(polygon.exterior.coords)


def visualize_dubins(use_arrows=True):
    """可视化Dubins曲线插值器的路径规划结果
    
    Args:
        use_arrows (bool): 是否使用箭头显示行进方向，默认为True
    """
    # 设置起始点的朝向角度和位置
    start_headings = np.arange(0.1, 2 * np.pi, 0.66)
    start_points = np.vstack((np.cos(start_headings), np.sin(start_headings))).T * 15 + np.array(
        [7.5, 7.5]
    )
    # 设置终点位置和朝向
    end_point = np.array([7.5, 7.5])
    end_heading = np.pi / 2
    # 设置车辆参数
    length = 4
    width = 1.8
    radius = 7.5

    # 创建Dubins曲线插值器
    my_dubins = Dubins(radius)

    # 创建图形对象
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 绘制终点车辆
    ax.add_patch(
        mpatches.Polygon(
            get_bbox(end_point, length, width, end_heading), fill=True, color="gray", alpha=0.5
        )
    )
    # 添加终点车辆标注
    ax.text(end_point[0], end_point[1] - 2, "end_vehicle", fontsize=12, ha='center')
    
    # 绘制终点车辆朝向箭头
    arrow_length = 3
    arrow_end = end_point + arrow_length * np.array([np.cos(end_heading), np.sin(end_heading)])
    ax.arrow(end_point[0], end_point[1], 
             arrow_end[0] - end_point[0], arrow_end[1] - end_point[1],
             head_width=0.8, head_length=1, fc='blue', ec='blue', zorder=10)

    # 绘制每个起始点的车辆和对应的Dubins路径
    for i, (start_point, start_heading) in enumerate(zip(start_points, start_headings)):
        # 绘制起始点车辆
        ax.add_patch(
            mpatches.Polygon(
                get_bbox(start_point, length, width, start_heading),
                fill=True,
                color="pink",
                alpha=0.5,
            )
        )
        # 添加起点车辆标注
        ax.text(start_point[0], start_point[1] - 2, f"start {i+1}", fontsize=10, ha='center')
        
        # 绘制起点车辆朝向箭头
        arrow_length = 3
        arrow_end = start_point + arrow_length * np.array([np.cos(start_heading), np.sin(start_heading)])
        ax.arrow(start_point[0], start_point[1], 
                 arrow_end[0] - start_point[0], arrow_end[1] - start_point[1],
                 head_width=0.8, head_length=1, fc='red', ec='red', zorder=10)
        
        # 计算并绘制Dubins路径
        path = my_dubins.get_curve(start_point, start_heading, end_point, end_heading)
        curve = path.curve
        
        if use_arrows:
            # 使用带箭头的线条显示行进方向
            for j in range(0, len(curve)-1, max(1, len(curve)//10)):  # 每隔一定间隔添加箭头
                ax.plot(curve[j:j+2, 0], curve[j:j+2, 1], "black")
                if j > 0 and j < len(curve) - 2:  # 避免在起点和终点附近添加箭头
                    mid_point = (curve[j] + curve[j+1]) / 2
                    direction = curve[j+1] - curve[j]
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        ax.arrow(mid_point[0], mid_point[1], 
                                 direction[0] * 0.5, direction[1] * 0.5,
                                 head_width=0.4, head_length=0.6, fc='black', ec='black', zorder=5)
        else:
            # 简单绘制路径线条
            ax.plot(curve[:, 0], curve[:, 1], "black")

    # 添加图例
    start_vehicle = mpatches.Patch(color='pink', alpha=0.5, label='start_vehicle')
    end_vehicle = mpatches.Patch(color='gray', alpha=0.5, label='end_vehicle')
    start_arrow = mlines.Line2D([], [], color='red', marker='>', linestyle='-', markersize=10, label='start_heading')
    end_arrow = mlines.Line2D([], [], color='blue', marker='>', linestyle='-', markersize=10, label='end_heading')
    
    legend_handles = [start_vehicle, end_vehicle, start_arrow, end_arrow]
    
    if use_arrows:
        path_arrow = mlines.Line2D([], [], color='black', marker='>', linestyle='-', markersize=10, label='path_arrow')
        legend_handles.append(path_arrow)
    else:
        path_line = mlines.Line2D([], [], color='black', linestyle='-', markersize=10, label='path_line')
        legend_handles.append(path_line)
    
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10)

    # 设置等比例显示
    ax.set_aspect("equal")
    # 添加标题
    mode_text = "arrows" if use_arrows else "simple"
    ax.set_title(f"Dubins path planning visualization ({mode_text} mode)", fontsize=14)
    # 保存图像
    filename = "Tactics2d_Tutorial/results/Dubins_arrows.png" if use_arrows else "Tactics2d_Tutorial/results/Dubins_simple.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # 设置mode
    mode = "simple"  # 可选值为"arrows"或"simple"
    if mode == "arrows":
        visualize_dubins(use_arrows=True)
    else:
        visualize_dubins(use_arrows=False)