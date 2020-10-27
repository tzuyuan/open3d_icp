import open3d as o3d
from utils import *

def color_icp(source, target, trans_init, down_sample=False):
    # check which radius to use for this dataset
    base_radius = compute_base_radius(source, target)
    voxel_radius = [base_radius]
    max_iter = [100]
    current_transformation = trans_init
    for scale in range(len(max_iter)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
 
        if down_sample:
            # down sample using 
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)
        else:
            # don't down sample
            source_down = source
            target_down = target

        if o3d.__version__=='0.8.0.0':
            # open3d 0.8.0
            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        else:
            # open3d 0.7.0
            o3d.geometry.estimate_normals(source_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            o3d.geometry.estimate_normals(target_down, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        current_transformation = result_icp.transformation

        # print("scale %i transformation" % (scale))
        # print(current_transformation)
        # draw_registration_result(source_down, target_down, current_transformation, True)

    return result_icp.transformation