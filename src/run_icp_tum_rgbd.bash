method="color_icp"
tum_folder="/media/sde1/tzuyuan/data/tum/"

for dataset in freiburg1_desk freiburg1_desk2 freiburg1_room freiburg1_360 freiburg1_teddy freiburg1_xyz freiburg1_rpy freiburg1_plant freiburg1_floor freiburg3_nostructure_texture_far freiburg3_nostructure_texture_near freiburg3_structure_notexture_far freiburg3_structure_notexture_near freiburg3_structure_texture_far freiburg3_structure_texture_near freiburg3_nostructure_notexture_far freiburg3_nostructure_notexture_near
do
    python3 src/main_rgbd_icp.py --seq $dataset --method $method #--tum_folder $tum_folder
done
