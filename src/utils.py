import open3d as o3d
import copy

def draw_registration_result(source, target, transformation, default_color):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if(default_color):
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
