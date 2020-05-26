import numpy as np
import argparse
import open3d as o3d
import copy
import glob
import cv2
import os
from kitti_handler import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Run Point to Plane ICP on Kitti Lidar Dataset')
    parser.add_argument('--seq', type=str, help='kitti sequence', default='05')
    parser.add_argument('--kitti_folder', type=str, help='folder path of kitti dataset', default='/media/justin/LaCie/data/kitti/sequences/')
    parser.add_argument('--method', type=str, help='the method of point cloud registration', default='color_icp')
    parser.add_argument('--output', type=str, help='output file path', default='../results/')
    parser.add_argument('--input_type', type=str, help='input type method', default='stereo')
    args = parser.parse_args()
    return args

def color_icp(source, target, trans_init):

    voxel_radius = [0.75, 0.3, 0.05]
    max_iter = [500, 300, 140]
    current_transformation = trans_init
    for scale in range(1):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])
 
        source_down = o3d.geometry.voxel_down_sample(source,radius)
        target_down = o3d.geometry.voxel_down_sample(target,radius)
        source_down = source
        target_down = target

        o3d.geometry.estimate_normals(source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*1.5, max_nn=30))
        o3d.geometry.estimate_normals(target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*1.5, max_nn=30))

        reg_color_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter))
        current_transformation = reg_color_icp.transformation
    # print("Transformation is:")
    # print(reg_color_icp.transformation)

    return reg_color_icp.transformation

def main():
    # formating print-out message
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=4)

    # we want to load kitti lidar bin file and run ICP in open3d
    args = parse_args()
    print("Running", args.method, "on kitti sequence", args.seq)
    
    if args.input_type == 'stereo':
        imgR_folder = os.path.join(args.kitti_folder, args.seq, "image_3")
        imgL_folder = os.path.join(args.kitti_folder, args.seq, "image_2")
        print("Loading files from\n", imgR_folder, "and \n", imgL_folder)
        file_path = sorted(glob.glob(imgR_folder+"/*.png"))
        imgL_files = sorted(glob.glob(imgL_folder+"/*.png"))
        # read calib file
        calib_pth = os.path.join(args.kitti_folder, args.seq, "cvo_calib.txt")
        calib_file = open(calib_pth, "r")
        calib_str = calib_file.readline().strip().split(" ")
        calib = [float(i) for i in calib_str]
        print("calib", calib)
    elif args.input_type == 'pcd':
        pcd_folder = os.path.join(args.kitti_folder, args.seq, "cvo_points_pcd")
        print("Loading files from", pcd_folder)
        file_path = sorted(glob.glob(pcd_folder+"/*.pcd"))
    else:
        print('Unavailable input type, please input stereo or pcd')
        return

    # Initialization of ICP
    threshold = 0.75
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    accum_tf = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

    # save to text file
    output_file_path = os.path.join(args.output, args.method, "icp_"+args.method+"_"+args.seq+".txt")
    transformation_file = open(output_file_path,"w") 
    np.savetxt(transformation_file, accum_tf[:3, :].reshape(1,-1))
    transformation_file.flush()
    
    # load the first frame
    if args.input_type == 'stereo':
        target = load_kitti_stereo(file_path[0], imgL_files[0], calib)
    elif args.input_type == 'pcd':
        target = load_kitti_from_pcd(file_path[0])

    # o3d.visualization.draw_geometries([target])

    for i, file_pth in enumerate(file_path[1:]):
        print("\n===== Aligning frame %i and %i =====" % (i, i+1))
        if args.input_type == 'stereo':
            source = load_kitti_stereo(file_pth, imgL_files[i+1], calib)
        elif args.input_type == 'pcd':
            source = load_kitti_from_pcd(file_pth)
        
        draw_registration_result(source, target, trans_init, True)

        print("Initial alignment")
        evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
        print(evaluation)

        transformation = []
        if args.method == "color_icp":
            transformation = color_icp(source,target,trans_init)
        else:
            print("Unavailable ICP method, please input color_icp.")
            return -1
        accum_tf = accum_tf@transformation

        print("Accumulated Transformation is:")
        print(accum_tf)

        # save file to the output file
        np.savetxt(transformation_file, accum_tf[:3, :].reshape(1,-1))
        transformation_file.flush()
        # update source point cloud
        target = copy.deepcopy(source)

    # close the file
    transformation_file.close()


if __name__ == '__main__':
    main()