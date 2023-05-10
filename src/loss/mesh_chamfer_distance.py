import torch
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds


def mesh_chamfer_distance(source_mesh, target_mesh, num_samples=5000):
    source_pc = sample_points_from_meshes(source_mesh, num_samples)
    target_pc = sample_points_from_meshes(target_mesh, num_samples)

    loss, _ = chamfer_distance(source_pc, target_pc)

    return loss

def mesh_distance(source_mesh, target_mesh, num_samples=5000):
    verts, normals = sample_points_from_meshes(source_mesh, num_samples, return_normals=True)
    source_pc = Pointclouds(points=verts, normals=normals)

    loss = point_mesh_face_distance(target_mesh, source_pc)

    return loss