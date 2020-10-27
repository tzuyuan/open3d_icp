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

def load_kitti_from_pcd(lidar_path):
    """ load point cloud from pcd file """
    return o3d.io.read_point_cloud(lidar_path)

def load_kitti_stereo(imgR_pth, imgL_pth, calib):
    """ load stereo images from kitti dataset and transform to point cloud """
    # read left and right image
    imgL = cv2.imread(imgL_pth)
    imgR = cv2.imread(imgR_pth)

    # set calibration aprameters [calib = fx, fy, cx, cy, baseline]
    fx = calib[0]
    fy = calib[1]
    cx = calib[2]
    cy = calib[3]
    baseline = calib[4]

    # compute disparity map
    disparity_map = getDisparity(imgL, imgR)
    h, w = disparity_map.shape

    # transform disparity to depth
    mask_map = disparity_map > 0
    depth_map = np.zeros_like(disparity_map)
    disparith_to_depth = baseline * fx * np.ones_like(disparity_map)

    with np.errstate(divide='ignore', invalid='ignore'):
        depth_map = np.true_divide(disparith_to_depth, disparity_map)
        depth_map[depth_map == np.inf] = 0
        depth_map = np.nan_to_num(depth_map)
    # depth_map = baseline * fx / disparity
    depth_image = o3d.geometry.Image(depth_map)
    color_image = o3d.io.read_image(imgL_pth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1000, convert_rgb_to_intensity=False)
    kitti_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    kitti_intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, kitti_intrinsic)
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
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity = (disparity-min_disp)/num_disp
    return disparity


def getParallaxMap(disparity):
    parallax_map = []
    x, y = disparity.shape
    
    for i in range(x):
        for j in range(y):
            parallax_map.append([j, i, disparity[i, j], 1])
            
    return np.array(parallax_map)


def reprojectImageTo3D(Q, parallax_mat):
    points_3D = Q @ parallax_mat.T
    points_3D /= points_3D[3, :]
    points_3D = points_3D[:3, :]
    points_3D = points_3D.T
    
    output = np.zeros((370, 1226, 3))
    i = 0
    j = 0
    
    for p in range(points_3D.shape[0]):
        output[i][j] = points_3D[p]
        j += 1
        
        if j >= 1226:
            j = 0
            i += 1
        
    
    print('points_3D shape after', output.shape)
    return output


def compute_normal(pcd):
    print("Compute the normal of the point cloud")
    o3d.geometry.estimate_normals(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                          max_nn=30))
    # o3d.visualization.draw_geometries([pcd])
    return pcd
