import numpy as np
import open3d as o3d
import cv2
import math
import os
from matplotlib import pyplot as plt

velo2cam = np.asarray([[0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

def load_tum_from_pcd(lidar_path):
    """ load point cloud from pcd file """
    return o3d.io.read_point_cloud(lidar_path)

def load_tum_rgbd(color_pth, depth_pth, calib):
    bgr_image = cv2.imread(color_pth)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)    
    depth_image = cv2.imread(depth_pth, 0)
    h, w = depth_image.shape

    fx = calib[0]
    fy = calib[1]
    cx = calib[2]
    cy = calib[3]
    scaling_factor = calib[4]

    output_points = []
    output_colors = []

    for row in range(h):
        for col in range(w):
            depth = depth_image[row, col]
            color = rgb_image[row, col, :]
            if depth!=0 and not math.isnan(depth):
                z = depth / scaling_factor
                x = (col - cx) * z / fx
                y = (row - cy) * z / fy
                output_points.append([x, y, z])
                output_colors.append(color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points) #numpy_points is your Nx3 cloud
    pcd.colors = o3d.utility.Vector3dVector(output_colors) #numpy_colors is an Nx3 matrix with the corresponding RGB colors
    
    print("checking point cloud value\n", output_points, "\n\ncehcking point cloud color\n", output_colors)

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
