method="color_icp"  
input_type="pcd"

for dataset in freiburg1_desk
do
    python3 main_rgbd_icp.py --seq $dataset --method $method --input_type $input_type
done
