method="color_icp"  

for dataset in 00 # 01 02 03 04 05 # 06 07 08 09 10
do
    python3 main_stereo_icp.py --seq $dataset --method $method
done