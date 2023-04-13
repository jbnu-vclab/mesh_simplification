import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes


def mesh_chamfer_distance(source_mesh, target_mesh, num_samples=5000):
    source_pc = sample_points_from_meshes(source_mesh, num_samples)
    target_pc = sample_points_from_meshes(target_mesh, num_samples)

    loss, _ = chamfer_distance(source_pc, target_pc)

    return loss