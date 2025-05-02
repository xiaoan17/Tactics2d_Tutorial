import os
import requests
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def download_file(url, filename):
    if not os.path.exists("data"):
        os.makedirs("data")
    filename = os.path.join("./data", filename)
    if not os.path.exists(filename):
        r = requests.get(url)
        if r.status_code == 200:
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024

            progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(filename, "wb") as f:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()


map_config_file = "map.config"

download_file(
    "https://raw.githubusercontent.com/WoodOxen/tactics2d/master/tactics2d/dataset_parser/map.config",
    map_config_file,
)


argoverse_json_file = "log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json"
argoverse_parquet_file = "scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet"

download_file(
    "https://raw.githubusercontent.com/SCP-CN-001/trajectory_dataset_support/main/trajectory_sample/Argoverse/train/0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca/log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json",
    argoverse_json_file,
)

download_file(
    "https://raw.githubusercontent.com/SCP-CN-001/trajectory_dataset_support/main/trajectory_sample/Argoverse/train/0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca/scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet",
argoverse_parquet_file,
)


DLP_agents_json_file = "DLP_DJI_0012_agents.json"
DLP_frames_json_file = "DLP_DJI_0012_frames.json"
DLP_instances_json_file = "DLP_DJI_0012_instances.json"
DLP_obstacles_json_file = "DLP_DJI_0012_obstacles.json"
DLP_osm_file = "DLP_map.osm"

# 前四个文件下载有问题，网站会进行检测。可以尝试手动下载。
download_file("https://datadryad.org/stash/downloads/file_stream/2654062", DLP_agents_json_file)
download_file("https://datadryad.org/stash/downloads/file_stream/2654066", DLP_frames_json_file)
download_file(
    "https://datadryad.org/stash/downloads/file_stream/2654081", DLP_instances_json_file
)
download_file(
    "https://datadryad.org/stash/downloads/file_stream/2654064", DLP_obstacles_json_file
)
download_file(
    "https://raw.githubusercontent.com/SCP-CN-001/trajectory_dataset_support/main/map/DLP/DLP.osm",
    DLP_osm_file,
)


interaction_osm_file = "DR_USA_Intersection_EP0.osm"
interaction_pedestrian_tracks_file = "pedestrian_tracks_000.csv"
interaction_vehicle_tracks_file = "vehicle_tracks_000.csv"

download_file(
    "https://raw.githubusercontent.com/SCP-CN-001/trajectory_dataset_support/main/map/INTERACTION/DR_USA_Intersection_EP0.osm",
    interaction_osm_file,
)
download_file(
    "https://raw.githubusercontent.com/SCP-CN-001/trajectory_dataset_support/main/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0/pedestrian_tracks_000.csv",
    interaction_pedestrian_tracks_file,
)
download_file(
    "https://raw.githubusercontent.com/SCP-CN-001/trajectory_dataset_support/main/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0/vehicle_tracks_000.csv",
    interaction_vehicle_tracks_file,
)


# 下载WOMD数据集
waymo_motion_data_file = "motion_data_one_scenario.tfrecord"
download_file(
    "https://github.com/SCP-CN-001/trajectory_dataset_support/raw/main/trajectory_sample/WOMD/motion_data_one_scenario.tfrecord",
    waymo_motion_data_file,
)
