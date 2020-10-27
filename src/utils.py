import open3d as o3d
import copy
import numpy as np
from pyquaternion import Quaternion

def draw_registration_result(source, target, transformation, default_color):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(default_color):
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def compute_base_radius(source, target):
    # check what radius to use for kdtree and normal estimation
    r1 = np.linalg.norm(source.get_max_bound() - source.get_min_bound())
    r2 = np.linalg.norm(target.get_max_bound() - target.get_min_bound())
    base_radius = min(r1,r2) 
    print("Base radius is : %f" % base_radius)
    
    return base_radius

def transformation_to_quaternion(rgb_name, accum_tf):
    rotation_matrix = accum_tf[:3, :3]
    quat = Quaternion(matrix=rotation_matrix)
    tum_result_form = np.array([rgb_name, accum_tf[0,3], accum_tf[1,3], accum_tf[2,3], quat.x, quat.y, quat.z, quat.w])

    return tum_result_form.reshape(1,-1)