import numpy as np
import open3d as o3d
import cv2
from matplotlib import pyplot as plt

velo2cam = np.asarray([[0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

def load_kitti_lidar(lidar_path):
    """ load point cloud from bin file """
    numpy_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    xyz_temp = numpy_points[:, :3]
    xyz_homogeneous = np.concatenate((xyz_temp.T, np.zeros((1,xyz_temp.shape[0]))))
    xyz_after_change_of_axis = velo2cam @ xyz_homogeneous
    xyz = xyz_after_change_of_axis[:3, :].T

    # transform from numpy array to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # compute normal
    pcd_with_normal = compute_normal(pcd)
    return pcd_with_normal

def load_kitti_from_pcd(pcd_path):
    """ load point cloud from pcd file """
    pcd = o3d.io.read_point_cloud(pcd_path)
    # o3d.io.write_point_cloud("kitti_point_cloud.pcd", pcd)
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def load_kitti_stereo(imgR_pth, imgL_pth, calib):
    """ load stereo images from kitti dataset and transform to point cloud """

    # set calibration aprameters [calib = fx, fy, cx, cy, baseline]
    fx = calib[0]
    fy = calib[1]
    cx = calib[2]
    cy = calib[3]
    baseline = calib[4]

    imgL = cv2.imread(imgL_pth)
    imgR = cv2.imread(imgR_pth)

    imgL_remove_sky = imgL[100:,:,:]
    imgR_remove_sky = imgR[100:,:,:]

    # disparity = getDisparity(imgL, imgR)
    disparity = getDisparity(imgL_remove_sky, imgR_remove_sky)
    h, w = disparity.shape

    # [x y z 1]^T = Q [u v d 1]^T
    Q = np.array([[1, 0, 0, -cx],
                  [0, 1, 0, -cy],
                  [0, 0, 0, fx],
                  [0, 0, 1/baseline, 0]])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    # colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    colors = cv2.cvtColor(imgL_remove_sky, cv2.COLOR_BGR2RGB)
    mask_map_zero = disparity > disparity.min()
    mask_map_z1 = points_3D[:, :, 2] < 40
    mask_map_z2 = points_3D[:, :, 2] > 0
    mask_map = mask_map_zero & mask_map_z1 & mask_map_z2


    output_points = points_3D[mask_map]
    output_colors = colors[mask_map].astype(np.float64) / 255


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points) #numpy_points is your Nx3 cloud
    pcd.colors = o3d.utility.Vector3dVector(output_colors) #numpy_colors is an Nx3 matrix with the corresponding RGB colors

    # o3d.io.write_point_cloud("kitti_point_cloud.pcd", pcd)
    # o3d.visualization.draw_geometries([pcd])

    return pcd

def getDisparity(imgL, imgR):
    # https://github.com/ShashwatNigam99/MR-assign-3-Stereo_Reconstruction_NonLinear_Optimization/blob/master/q1/Q1.ipynb
    
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    window_size = 5
    min_disp = -39
    num_disp = 144
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        disp12MaxDiff = 1,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    
        P2=32 * 3 * window_size ** 2,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        preFilterCap=63
        )
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0 #the map is a 16-bit signed single-channel image
    # disparity = (disparity-min_disp)/num_disp
    return disparity


def compute_normal(pcd):
    print("Compute the normal of the point cloud")
    o3d.geometry.estimate_normals(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                          max_nn=30))
    # o3d.visualization.draw_geometries([pcd])
    return pcd
