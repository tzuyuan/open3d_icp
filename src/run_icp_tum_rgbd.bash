method="color_icp"  
input_type="rgbd"
tum_folder="/media/sde1/tzuyuan/data/tum/"

for dataset in freiburg1_desk freiburg1_desk2 freiburg1_room freiburg1_360 freiburg1_teddy freiburg1_xyz freiburg1_rpy freiburg1_plant freiburg1_floor
do
    python3 main_rgbd_icp.py --seq $dataset --method $method --input_type $input_type --tum_folder $tum_folder
done
