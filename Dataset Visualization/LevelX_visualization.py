import os
import warnings
import numpy as np
import json
warnings.filterwarnings("ignore")
from tactics2d.dataset_parser import LevelXParser
from tactics2d.map.parser import OSMParser
from tactics2d.traffic.scenario_display import ScenarioDisplay
import xml.etree.ElementTree as ET

def demo_levelx(
    dataset,
    trajectory_file,
    trajectory_folder,
    map_folder,
    map_config_path,
    xlim=None,
    ylim=None,
    time_range=(0, 10000),
    fps=25,
    interval=40,
    export_to=None,
):
    """
    可视化LevelX数据集场景
    
    参数:
        dataset: 数据集名称，如"highD"
        file: 轨迹数据文件名
        folder: 数据文件所在文件夹
        xlim: x轴显示范围，默认为[0, 300]
        ylim: y轴显示范围，默认为[-25, 5]
        time_range: 时间范围，默认为(0, 10000)
        fps: 导出视频的帧率，默认为25
        interval: 帧间隔（毫秒），默认为None（根据fps自动计算）
        export_to: 导出文件路径，默认为None（不导出）
    """
    # 计算帧间隔
    if interval is None:
        interval = int(1000 / fps)
    
    # 初始化LevelX数据解析器
    dataset_parser = LevelXParser(dataset)
    
    # 解析轨迹数据
    trajectories, actual_time_range = dataset_parser.parse_trajectory(trajectory_file, trajectory_folder, time_range)
    
    # 根据时间范围和间隔生成帧序列
    frames = np.arange(actual_time_range[0], actual_time_range[1], interval)
    
    # 获取地图信息
    location = dataset_parser.get_location(trajectory_file, trajectory_folder)
    map_name = f"{dataset}_{location}"
    
    # 解析地图数据
    with open(map_config_path, "r") as f:
        map_configs = json.load(f)
    map_config = map_configs[map_name]


    map_path = os.path.join(map_folder, map_config["osm_path"])
    map_root = ET.parse(map_path).getroot()
    lanelet2_parser = OSMParser(lanelet2=True)

    map_ = lanelet2_parser.parse(
        map_path, map_config
    )

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

    map_config_path = r"data/map.config"
    map_folder = r"/home/data1/Anbc_Save/研究用数据集/2024-12_LevelX-map"
    # 此处以highD为例，还有其他的数据集，替换文件路径即可。
    dataset_name = "exiD"  # 可选: "highD", "inD", "rounD", "exiD"

    if dataset_name == "highD":
        trajectory_file = "01_tracks.csv"
        trajectory_folder = r"/home/data1/Anbc_Save/研究用数据集/2022-05_highD_old/data"
        xlim=[0, 300]
        ylim=[-35, 5]
    elif dataset_name == "inD":
        trajectory_file = "00_tracks.csv"
        trajectory_folder = r"/home/data1/Anbc_Save/研究用数据集/2024-10_inD/data"
        xlim=[50, 200]
        ylim=[-125, -25]
    elif dataset_name == "rounD":
        trajectory_file = "03_tracks.csv"
        trajectory_folder = r"/home/data1/Anbc_Save/研究用数据集/2024-10_rounD/data"
        xlim=[0, 150]
        ylim=[-100, 5]
        # xlim=[50, 200]
        # ylim=[-145, -25]
    elif dataset_name == "exiD":
        trajectory_file = "01_tracks.csv"
        trajectory_folder = r"/home/data1/Anbc_Save/研究用数据集/2024-10_exiD-dataset-v2.0/data"
        xlim=[300, 600]
        ylim=[-500, -100]

    # 可视化LevelX样例场景
    demo_levelx(
        dataset=dataset_name,
        trajectory_file=trajectory_file,
        trajectory_folder=trajectory_folder,
        map_folder=map_folder,
        map_config_path=map_config_path,
        xlim=xlim,  # 设置x轴显示范围
        ylim=ylim,  # 设置y轴显示范围
        fps=25,
        export_to=f"results/levelx_{dataset_name}_sample_scenario.mp4",  # 导出路径
    )
