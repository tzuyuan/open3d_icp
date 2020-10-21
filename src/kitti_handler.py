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
    imgL = cv2.imread(imgL_pth)
    imgR = cv2.imread(imgR_pth)
    # imgL_gray = cv2.imread(imgL_pth,0)
    # imgR_gray = cv2.imread(imgR_pth,0)

    # # Set disparity parameters
    # # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
    # # https://stackoverflow.com/questions/45325795/point-cloud-from-kitti-stereo-images

    # window_size = 9
    # minDisparity = 1
    # stereo = cv2.StereoSGBM_create(
    #     blockSize=10,
    #     numDisparities=64,
    #     preFilterCap=10,
    #     minDisparity=minDisparity,
    #     P1=4 * 3 * window_size ** 2,
    #     P2=32 * 3 * window_size ** 2
    # )

    # # Compute disparity map
    # disparity_map = stereo.compute(imgL_gray, imgR_gray)
    # disparity_map = disparity_map.astype(float)

    # # Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
    # plt.imshow(disparity_map,'gray')
    # plt.show()

    # # mask_map = disparity_map > 0
    # depth_map = np.zeros_like(disparity_map)
    # disparith_to_depth = calib[4] * calib[0] * np.ones_like(disparity_map)

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     depth_map = np.true_divide(disparith_to_depth, disparity_map)
    #     depth_map[depth_map == np.inf] = 0
    #     depth_map = np.nan_to_num(depth_map)

    # depth_scale = 1000.0
    # depth_map = depth_map / depth_scale

    # # creating RGBD image
    # depth_map = np.asarray(depth_map).astype(np.float32)
    # depth_image = o3d.geometry.Image(depth_map)
    # rgb_image = o3d.geometry.Image(imgL)

    # rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(rgb_image, depth_image)

    # intrinsic = o3d.camera.PinholeCameraIntrinsic(imgL_gray.shape[1], imgL_gray.shape[0], calib[0], calib[1], calib[2], calib[3])
    # print('intrinsic\n', intrinsic.intrinsic_matrix)
    # pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)

    disparity = getDisparity(imgL, imgR)

    w = disparity.shape[0]
    l = disparity.shape[1]

    K = np.array([[calib[0], 0.0,      calib[2]], 
                    [0.0,    calib[1], calib[3]], 
                    [0.0,    0.0,          1.0]])

    Q = np.array([[1, 0, 0, -l/2],
                [0, 1, 0, -w/2],
                [0, 0, 0, calib[0]],
                [0, 0, -1/calib[4], 0]])

    parallax_map = getParallaxMap(disparity)
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    # points_3D = reprojectImageTo3D(Q, parallax_map)

    mask_map_zero = disparity > disparity.min()
    # print('points_3D', points_3D)
    mask_map_z = points_3D[:, :, 2] < 50
    mask_map = mask_map_zero & mask_map_z

    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points) #numpy_points is your Nx3 cloud
    pcd.colors = o3d.utility.Vector3dVector(output_colors) #numpy_colors is an Nx3 matrix with the corresponding RGB colors

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
