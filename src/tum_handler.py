import numpy as np
import open3d as o3d
import cv2
# import math
import os
from matplotlib import pyplot as plt

def load_tum_from_pcd(lidar_path):
    """ load point cloud from pcd file """
    return o3d.io.read_point_cloud(lidar_path)

def load_tum_rgbd(color_pth, depth_pth, calib):
    print("Load TUM color and depth image to pointcloud")
    depth_image = cv2.imread(depth_pth, 0)
    h, w = depth_image.shape

    fx = calib[0]
    fy = calib[1]
    cx = calib[2]
    cy = calib[3]
    scaling_factor = calib[4]
    
    color_raw = o3d.io.read_image(color_pth)
    depth_raw = o3d.io.read_image(depth_pth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=scaling_factor, convert_rgb_to_intensity=False)
    tum_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    tum_intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, tum_intrinsic)
    # o3d.visualization.draw_geometries([pcd])
    
    return pcd
    
def load_file_name(folder_pth):
    assoc_pth = os.path.join(folder_pth, "assoc.txt")
    rgb_names = []
    rgb_paths = []
    depth_names = []
    depth_paths = []
    
    assoc_file = open(assoc_pth, "r")
    for line in assoc_file:
        assoc_list = line.split()
        rgb_names.append(assoc_list[0])
        rgb_paths.append(os.path.join(folder_pth, assoc_list[1]))
        depth_names.append(os.path.join(folder_pth, assoc_list[2]))
        depth_paths.append(os.path.join(folder_pth, assoc_list[3]))
    assoc_file.close()

    return rgb_names, rgb_paths, depth_names, depth_paths
