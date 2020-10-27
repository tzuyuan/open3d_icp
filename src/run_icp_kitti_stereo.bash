method="color_icp"
kitti_folder="/home/rzh/datasets/kitti/sequences/"

for dataset in 00 01 02 03 04 05 # 06 07 08 09 10
do
    python3 src/main_stereo_icp.py --seq $dataset --method $method #--kitti_folder $kitti_folder
done
