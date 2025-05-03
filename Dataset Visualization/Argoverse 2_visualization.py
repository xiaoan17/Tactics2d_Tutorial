import numpy as np
from tactics2d.dataset_parser import ArgoverseParser
from tactics2d.traffic.scenario_display import ScenarioDisplay


def demo_argoverse(
    file, map_file, folder, xlim=None, ylim=None, fps=10, interval=100, export_to=None
):
    """
    可视化Argoverse数据集中的场景
    
    参数:
        file: 轨迹数据文件名
        map_file: 地图数据文件名
        folder: 数据文件所在文件夹
        xlim: x轴显示范围，默认为None（自动确定）
        ylim: y轴显示范围，默认为None（自动确定）
        fps: 导出视频的帧率，默认为10
        interval: 帧间隔（毫秒），默认为100
        export_to: 导出文件路径，默认为None（不导出）
    """
    # 初始化Argoverse数据解析器
    dataset_parser = ArgoverseParser()
    
    # 解析轨迹数据
    trajectories, actual_time_range = dataset_parser.parse_trajectory(file, folder)
    
    # 根据时间范围和间隔生成帧序列
    frames = np.arange(actual_time_range[0], actual_time_range[1], interval)
    
    # 解析地图数据
    map_ = dataset_parser.parse_map(map_file, folder)

    # 初始化场景显示器
    scenario_display = ScenarioDisplay()
    scenario_display.reset()
    
    # 设置显示区域和图像尺寸
    if xlim is None or ylim is None:
        # 如果未指定显示范围，使用默认设置
        ax_settings = {"aspect": "equal"}
        fig_size = (4, 3)
    else:
        # 根据指定的显示范围设置图像尺寸，保持比例一致
        ax_settings = {"aspect": "equal", "xlim": xlim, "ylim": ylim}
        fig_size = (4, round((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) * 4, 1))

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
    # 可视化Argoverse 2样例场景
    demo_argoverse(
        file="scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet",
    map_file="log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json",
    folder="./data",
    xlim=[1800, 2100],  # 设置x轴显示范围
    ylim=[500, 750],    # 设置y轴显示范围
    export_to="results/argoverse_v2_sample_scenario.mp4",  # 导出路径
    )
