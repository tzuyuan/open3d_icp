import numpy as np
import argparse
import open3d as o3d
import copy
import glob
import cv2
import os
from tum_handler import *
from utils import *
from icp_methods import *

def parse_args():
    parser = argparse.ArgumentParser(description='Run Color ICP on TUM RGB-D Dataset')
    parser.add_argument('--seq', type=str, help='tum sequence', default='')
    parser.add_argument('--tum_folder', type=str, help='folder path of tum dataset', default='/home/cel/CURLY/code/DockerFolder/data/tum/')
    parser.add_argument('--method', type=str, help='the method of point cloud registration', default='color_icp')
    parser.add_argument('--output', type=str, help='output file path', default='results/')
    parser.add_argument('--input_type', type=str, help='input type method', default='rgbd')
    args = parser.parse_args()
    return args

def main():
    # formating print-out message
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=4)

    # we want to load RGBD file and run ICP in open3d
    args = parse_args()
    print("Running", args.method, "on RGBD sequence", args.seq)
    
    if args.input_type == 'rgbd':
        # get associate names and paths
        seq_pth = os.path.join(args.tum_folder, args.seq)
        rgb_names, file_path, depth_names, depth_paths = load_file_name(seq_pth)
        
        # get color and depth image folder 
        color_folder = os.path.join(args.tum_folder, args.seq, "rgb")
        depth_folder = os.path.join(args.tum_folder, args.seq, "depth")
        print("Loading files from\n", color_folder, "and \n", depth_folder)
        
        # read calib file
        calib_pth = os.path.join(args.tum_folder, args.seq, "cvo_calib.txt")
        calib_file = open(calib_pth, "r")
        calib_str = calib_file.readline().strip().split(" ")
        calib = [float(i) for i in calib_str]
        print("calib", calib)
    elif args.input_type == 'pcd':
        pcd_folder = os.path.join(args.tum_folder, args.seq, "pcd_full")
        print("Loading files from", pcd_folder)
        file_path = sorted(glob.glob(pcd_folder+"/*.pcd"))
    else:
        print('Unavailable input type, please input rgbd or pcd')
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
    output_file_folder = os.path.join(args.output, args.method)
    # create result folder if not exists
    if not os.path.exists(output_file_folder):
        os.makedirs(output_file_folder)
    output_file_path = os.path.join(args.output, args.method, "icp_"+args.method+"_"+args.seq+".txt")
    transformation_file = open(output_file_path,"w") 
    np.savetxt(transformation_file, transformation_to_quaternion(rgb_names[0], accum_tf), fmt="%s %s %s %s %s %s %s %s")
    transformation_file.flush()
    
    # load the first frame
    if args.input_type == 'rgbd':
        target = load_tum_rgbd(file_path[0], depth_paths[0], calib)
    elif args.input_type == 'pcd':
        target = load_tum_from_pcd(file_path[0])

    # o3d.visualization.draw_geometries([target])

    for i, file_pth in enumerate(file_path[1:]):
        print("\n===== Aligning frame %i and %i =====" % (i, i+1))
        if args.input_type == 'rgbd':
            source = load_tum_rgbd(file_pth, depth_paths[i+1], calib)
        elif args.input_type == 'pcd':
            source = load_tum_from_pcd(file_pth)
        
        # print("initial alignment for frame %i and %i" % (i, i+1))
        # draw_registration_result(source, target, trans_init, True)

        transformation = []
        if args.method == "color_icp":
            transformation = color_icp(source, target, trans_init)
            # print("transformation after color icp")
            # draw_registration_result(source, target, transformation, True)
            evaluation = o3d.registration.evaluate_registration(source, target, threshold, transformation)
            print("evaluation:", evaluation)
        else:
            print("Unavailable ICP method, please input color_icp.")
            return -1
        accum_tf = accum_tf@transformation

        print("Accumulated Transformation is:")
        print(accum_tf)

        # save file to the output file
        np.savetxt(transformation_file, transformation_to_quaternion(rgb_names[i+1], accum_tf), fmt="%s %s %s %s %s %s %s %s")
        transformation_file.flush()
        # update source point cloud
        target = copy.deepcopy(source)

    # close the file
    transformation_file.close()


if __name__ == '__main__':
    main()
