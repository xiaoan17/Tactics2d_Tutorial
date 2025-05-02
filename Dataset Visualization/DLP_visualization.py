import numpy as np
import os
import json
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

from tactics2d.dataset_parser import DLPParser
from tactics2d.map.parser import OSMParser
from tactics2d.traffic.scenario_display import ScenarioDisplay


def demo_dlp(
    file, map_file, folder, xlim=None, ylim=None, fps=25, interval=None, time_range=(0, 20000), export_to=None
):
    """
    可视化DLP数据集中的场景
    
    参数:
        file: 轨迹数据文件名
        map_file: 地图数据文件名
        folder: 数据文件所在文件夹
        xlim: x轴显示范围，默认为None（自动确定）
        ylim: y轴显示范围，默认为None（自动确定）
        fps: 导出视频的帧率，默认为25
        interval: 帧间隔（毫秒），默认为根据fps计算
        time_range: 时间范围，默认为(0, 20000)
        export_to: 导出文件路径，默认为None（不导出）
    """
    # 计算帧间隔
    if interval is None:
        interval = int(1000 / fps)
    
    # 初始化DLP数据解析器
    dataset_parser = DLPParser()
    
    # 解析轨迹数据
    trajectories, actual_time_range = dataset_parser.parse_trajectory(
        file, folder, time_range
    )
    
    # 根据时间范围和间隔生成帧序列
    frames = np.arange(actual_time_range[0], actual_time_range[1], interval)
    
    # 解析地图数据
    map_config_path = os.path.join(folder, "map.config")
    with open(map_config_path, "r") as f:
        map_configs = json.load(f)
    
    map_config = map_configs["DLP"]
    map_path = os.path.join(folder, map_file)
    map_root = ET.parse(map_path).getroot()
    
    lanelet2_parser = OSMParser(lanelet2=True)
    map_ = lanelet2_parser.parse(
        map_root, map_config["project_rule"], map_config["gps_origin"], map_config
    )

    # 初始化场景显示器
    scenario_display = ScenarioDisplay()
    scenario_display.reset()
    
    # 设置显示区域和图像尺寸
    if xlim is None or ylim is None:
        # 如果未指定显示范围，使用默认设置
        ax_settings = {"aspect": "equal"}
        fig_size = (5, 3)
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

# 可视化DLP样例场景

# TODO: 只用到了file文件，同时parser中文件名已经写死，不要随意修改文件名。
demo_dlp(
    file="DJI_0012_agents.json",
    map_file="DLP_map.osm",
    folder="./data",
    xlim=[-5, 145],  # 设置x轴显示范围
    ylim=[0, 80],    # 设置y轴显示范围
    export_to="results/DLP_sample.mp4",  # 导出路径
)
