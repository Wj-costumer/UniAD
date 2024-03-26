# visualize.py
import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]

NameMapping = {
    'movable_object.barrier': 'barrier', 
    'vehicle.bicycle': 'bicycle', 
    'vehicle.bus.bendy': 'bus', 
    'vehicle.bus.rigid': 'bus', 
    'vehicle.car': 'car', 
    'vehicle.construction': 'construction_vehicle', 
    'vehicle.motorcycle': 'motorcycle', 
    'human.pedestrian.adult': 'pedestrian', 
    'human.pedestrian.child': 'pedestrian', 
    'human.pedestrian.construction_worker': 'pedestrian', 
    'human.pedestrian.police_officer': 'pedestrian', 
    'movable_object.trafficcone': 'traffic_cone', 
    'vehicle.trailer': 'trailer', 
    'vehicle.truck': 'truck'
    }

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "tricycle": (220, 20, 60),  # 相比原版 mmdet3d 的 visualize 增加 tricycle
    "cyclist": (220, 20, 60)  # 相比原版 mmdet3d 的 visualize 增加 cyclist
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

IMGS_PATH = "./projects/UniAD/data/nuscenes"
OUTPUT_PATH = "./outputs"
pc_range = 100
image_size = 1024

def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

    
def combine_all(images, combine_save_path):
    """将 6 个视角的图片和 bev视角下的 lidar 进行拼接

    :param img_path_dict: 每个视角的 img 图片路径 && bev 视角下 lidar 的图片路径

    |----------------------------------------------  | -------------
    |cam_front_left | cam_front  | cam_front_right   |
    |-------------- | ---------  | ---------------   |  lidar_bev
    |cam_back_left  | cam_back   | cam_back_right    |  
    |----------------------------------------------  | -------------

    """
    cam_front = images["CAM_FRONT"]
    cam_front_left = images["CAM_FRONT_LEFT"]
    cam_front_right = images["CAM_FRONT_RIGHT"]
    cam_back = images["CAM_BACK"]
    cam_back_left = images["CAM_BACK_LEFT"]
    cam_back_right = images["CAM_BACK_RIGHT"]
    # merge img
    front_combined = cv2.hconcat([cam_front_left, cam_front, cam_front_right])
    back_combined = cv2.hconcat([cam_back_right, cam_back, cam_back_left])
    back_combined = cv2.flip(back_combined, 1)  # 左右翻转
    img_combined = cv2.vconcat([front_combined, back_combined])
    # 读 lidar
    lidar_bev = images["lidar"]
    # img_combined 等比例缩小
    target_height = lidar_bev.shape[0]
    scale_factor = target_height / img_combined.shape[0]
    target_width = int(img_combined.shape[1] * scale_factor)
    img_combined = cv2.resize(img_combined, (target_width, target_height))
    # merge all
    merge_image = cv2.hconcat([img_combined, lidar_bev])
    # 保存图片
    cv2.imwrite(combine_save_path, merge_image)
    
    return merge_image

def get_matrix(calibrated_data, inverse = False): 
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

def visualize_lidar_points(
    lidar_file_path: str,
    ego2lidar,
    pc_range: int,
    image_size: int
) -> np.array:
    mmcv.check_file_exist(lidar_file_path)
    points = np.frombuffer(open(lidar_file_path, 'rb').read(), dtype=np.float32)
    points = points.reshape(-1, 5)[:, :3] # x, y, z, intensity, ring_idx
    points = np.concatenate([points, np.ones((len(points), 1))], axis = 1)
    points = points @ ego2lidar.T # convert to lidar
    pc_range = 100
    points[:, :2] /= pc_range
    image_size = 1024
    # 将pts缩放成图像大小，并平移到中心
    points[:, :2] = points[:, :2] * image_size / 2 + image_size / 2
    
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for ix, iy, iz in points[:, :3]:
        if 0 <= ix < image_size and 0 <= iy <image_size:
            image[int(ix), int(iy)] = 255, 255, 255 
    return image

def visualize_2D_boxes(
    nusc,
    image, 
    boxes,
    pc_range: int,
    image_size: int 
    ) -> np.array:
    # vsiualize 2D boxes
    for box in boxes:
        x, y, z = box.center
        w, h, l = box.wlh
        corners = [(x + w/2, y + h/2), (x + w/2, y - h/2), (x - w/2, y - h/2), (x - w/2, y + h/2)]
        corners = [(int(x / pc_range * image_size / 2 + image_size / 2), int(y / pc_range * image_size / 2 + image_size / 2)) for x, y in corners]
        annotation = nusc.get('sample_annotation', box.token)
        if box.name in NameMapping:
            name = NameMapping[box.name]
            color = OBJECT_PALETTE[name]
            for i in range(4):
                cv2.line(image, corners[i], corners[(i + 1) % 4], color, 2)
    return image

def visualize_3D_boxes(
    nusc,
    image_path,
    boxes,
    camera_intrinsic,
    pc_range,
    image_size
) -> np.array:
    
    image_cam = cv2.imread(image_path)
    for box in boxes:
        corners = box.corners().T # 8 x 3
        corners = np.concatenate([corners, np.ones((len(corners), 1))], axis = 1) @ camera_intrinsic.T # 8 x 4
        corners[:, :2] /= corners[:, [2]]
        if box.name in NameMapping:
            name = NameMapping[box.name]
            color = OBJECT_PALETTE[name]
            for start, end in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]:
                cv2.line(image_cam, tuple(corners[start][:2].astype(np.int)), tuple(corners[end][:2].astype(np.int)), color, thickness=2)
    return image_cam

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot=IMGS_PATH, verbose=True)
    frame_idx = 0
    scene_idx = 0
    for sample in nusc.sample:
        if frame_idx == 0:
            last_scene_token = sample['scene_token']
        if sample['scene_token'] != last_scene_token:
            frame_idx = 0 # next scene frame start from 0
            scene_idx += 1
            last_scene_token = sample['scene_token'] 
        frame_idx += 1
        mmcv.mkdir_or_exist(OUTPUT_PATH + f'/scene_{scene_idx}') 
        images = {'lidar': None}
        lidar_token = sample['data']['LIDAR_TOP'] 
        lidar_sample_data = nusc.get('sample_data', lidar_token)
        lidar_calibrated_data = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
        ego2lidar = get_matrix(lidar_calibrated_data, inverse=True)
        ego_calibrated_data_l = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        ego2global = get_matrix(ego_calibrated_data_l, inverse=False)
        # boxes in lidar coordinate system
        lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token) # get_sample_data函数会将box投影到输入的token对应的传感器坐标系下
        # visualize lidar points
        breakpoint()
        image_2D = visualize_lidar_points(lidar_path, ego2lidar, pc_range, image_size)
        # visualize 2D boxes
        image_2D = visualize_2D_boxes(nusc, image_2D, boxes_lidar, pc_range, image_size)
        
        images["lidar"] = image_2D
        
        # visualize 3D on camera images
        cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
        for camera in cameras:
            camera_token = sample["data"][camera]
            camera_sample_data = nusc.get('sample_data', camera_token)
            image_path, boxes_cam, camera_intrinsic_matrix = nusc.get_sample_data(camera_token) # 3D boxes in camera coordinate system
            camera_intrinsic = np.eye(4)
            camera_intrinsic[:3, :3] = camera_intrinsic_matrix
            image_3D = visualize_3D_boxes(nusc, image_path, boxes_cam, camera_intrinsic, pc_range, image_size)
        
            images[camera] = image_3D
        combine_all(images, OUTPUT_PATH + f'/scene_{scene_idx}/cam_{frame_idx}.png')
        
