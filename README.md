# ROS node YOLOV8 seg on rk3588

## Overview

a pkg using gpu for yolo seg

![](https://raw.githubusercontent.com/TonyTnT/picgo/main/202405241117027.gif)

comparing to inference on rk3588 (infer time < 20ms), the infer time < 2ms


## Requirements

- gpu, e.g. 4090
- any rgbd cam topic


## Installation
### Preparation

1. Install Ultralytics
2. Prepare ROS IMAGE TOPICS


### About pkg
1. Clone this repository into your workspace.
2. Build the package using `colcon build --packages-select yolov8_seg `.
3. Source the setup file: `source install/setup.bash`.


to keep the time-accurated position of the objects, using ros pkg `message_filters` to sync `color/image_raw` and `depth/image_raw`

though using image from `web video server` (websocket) seems have lower latency 


## Usage

```
ros2 run yolov8_seg infer_node
```