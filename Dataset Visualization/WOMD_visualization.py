import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from tactics2d.dataset_parser import WOMDParser
from tactics2d.traffic.scenario_display import ScenarioDisplay

def demo_womd(
    file="motion_data_one_scenario.tfrecord",
    folder="./data",
    xlim=[-7890, -7710],
    ylim=[-6775, -6600],
    fps=20,
    interval=None,
    export_to=None,
):
    """
    可视化Waymo Open Motion数据集场景
    
    参数:
        file: 轨迹数据文件名
        folder: 数据文件所在文件夹
        xlim: x轴显示范围，默认为[-7890, -7710]
        ylim: y轴显示范围，默认为[-6775, -6600]
        fps: 导出视频的帧率，默认为20
        interval: 帧间隔（毫秒），默认为None（根据fps自动计算）
        export_to: 导出文件路径，默认为None（不导出）
    """
    # 计算帧间隔
    if interval is None:
        interval = int(1000 / fps)
    
    # 初始化WOMD数据解析器
    dataset_parser = WOMDParser()
    
    # 解析轨迹数据
    trajectories, actual_time_range = dataset_parser.parse_trajectory(file=file, folder=folder)
    
    # 根据时间范围和间隔生成帧序列
    frames = np.arange(actual_time_range[0], actual_time_range[-1], interval)
    
    # 解析地图数据
    map_ = dataset_parser.parse_map(file=file, folder=folder)

    # 初始化场景显示器
    scenario_display = ScenarioDisplay()
    scenario_display.reset()
    
    # 设置显示区域和图像尺寸
    if xlim is None or ylim is None:
        # 如果未指定显示范围，使用默认设置
        ax_settings = {"aspect": "equal"}
        fig_size = (5, 5)
    else:
        # 根据指定的显示范围设置图像尺寸，保持比例一致
        ax_settings = {"aspect": "equal", "xlim": xlim, "ylim": ylim}
        fig_size = (5, round((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) * 5, 1))

    # 生成动画
    animation = scenario_display.display(
        trajectories, map_, interval, frames, fig_size, **ax_settings
    )

    # 处理导出选项
    if export_to is None:
        return animation
    else:
        # 导出为视频文件
        animation.save(filename=export_to, writer="ffmpeg", fps=fps, dpi=300)

if __name__ == "__main__":
    # 可视化WOMD样例场景
    demo_womd(
        file="motion_data_one_scenario.tfrecord",
        folder="./data",
        xlim=[-7890, -7710],  # 设置x轴显示范围
        ylim=[-6775, -6600],  # 设置y轴显示范围
        fps=20,
        export_to="results/womd_sample_scenario.mp4",  # 导出路径
    )