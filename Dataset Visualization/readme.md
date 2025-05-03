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

### 可能遇到的问题
- **inD的osm**：会提示multipolygon、左右车道错误等信息，可以注释掉出错的osm部分来处理。
- **exiD的osm**：存在节点和道路元素的id复用，需要校对并避免重复，手动替换掉重复的id。仔细查看原因，是部分node节点的数据与road的id重复了，这导致存在上述的问题。
- **rounD的osm**：会提示没有oneway这个参数。在class的解析中，并没有过多的考虑这个参数。osm文件命名与原始文档的对应关系。
  - round 0 -> location0.osm
  - round 1 -> 暂无
  - round 2 -> 暂无
