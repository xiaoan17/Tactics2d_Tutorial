# 数据归类

## Argoverse v2
- file: "scenario_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.parquet"
- map_file: "log_map_archive_0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca.json"

## DLP
- file: "DJI_0012_agents.json"
- map_file: "DLP_map.osm"

## INTERACTION
- file: "vehicle_tracks_000.csv"
- map_file: "DR_USA_Intersection_EP0.osm"

## WOMD
- file: "motion_data_one_scenario.tfrecord"

## LevelX
**警告**：记得要调整不同的地图对应的x，y的坐标范围，这个对示意图的影响比较大。
路网与轨迹文件的对应关系，请查看作者提供的tactics2d的map.config文件。

### 数据集对应关系
- **highD数据集**：osm文件可以在作者提供的原始文件中查看。
- **inD、rounD、exiD**：需要从官方提供的osm文件中查询。
