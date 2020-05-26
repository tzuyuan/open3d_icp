import numpy as np
import argparse
import open3d as o3d
import copy
import glob
# import cv2
import os
from kitti_handler import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Run Point to Plane ICP on Kitti Lidar Dataset')
    parser.add_argument('--seq', type=str, help='kitti sequence', default='05')
    parser.add_argument('--kitti_folder', type=str, help='folder path of kitti dataset', default='/media/justin/LaCie/data/kitti/sequences/')
    parser.add_argument('--method', type=str, help='the method of point cloud registration', default='point_to_point')
    parser.add_argument('--output', type=str, help='output file path', default='../results/')
    args = parser.parse_args()
    return args



def point_to_plane_icp(source, target, threshold, trans_init):
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
    print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation, True)
    return reg_p2l.transformation

def point_to_point_icp(source, target, threshold, trans_init):
    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
    print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation, True)
    return reg_p2p.transformation

def main():
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=4)
    # we want to load kitti lidar bin file and run ICP in open3d
    args = parse_args()
    print("Running", args.method, "on kitti sequence", args.seq)    
    lidar_folder = os.path.join(args.kitti_folder, args.seq, "velodyne")
    print("Loading files from", lidar_folder)
    lidar_files = sorted(glob.glob(lidar_folder+"/*.bin"))

    # Initialization of ICP
    threshold = 0.75
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                          [-0.139, 0.967, -0.215, 0.7],
    #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
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
    target = load_kitti_lidar(lidar_files[0])    

    for i, lidar_path in enumerate(lidar_files[1:]):
        print("\n===== Aligning frame %i and %i =====" % (i, i+1))
        source = load_kitti_lidar(lidar_path)
        # draw_registration_result(source, target, trans_init, True)

        print("Initial alignment")
        evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
        print(evaluation)

        transformation = []
        if args.method == "point_to_point":
            transformation = point_to_point_icp(source, target, threshold, trans_init)
        elif args.method == "point_to_plane":
            transformation = point_to_plane_icp(source, target, threshold, trans_init)
        else:
            print("Unavailable ICP method, please input point_to_point or point_to_plane.")
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

