import torch
import numpy as np
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from src.ops.sample_points_from_meshes import barycentric_sampling_from_meshes

def mesh_chamfer_distance(source_mesh, target_mesh, num_samples=5000, sampling_method='random', norm=2):
    if sampling_method == 'random':
        source_pc = sample_points_from_meshes(source_mesh, num_samples)
        target_pc = sample_points_from_meshes(target_mesh, num_samples)
    if sampling_method == 'barycentric':
        source_pc = barycentric_sampling_from_meshes(source_mesh)
        target_pc = barycentric_sampling_from_meshes(target_mesh)

    spcl = Pointclouds(source_pc)
    tpcl = Pointclouds(target_pc)

    loss, _ = chamfer_distance(source_pc, target_pc, norm=norm)

    return loss, spcl, tpcl

def point_to_mesh_distance(source_mesh, target_mesh, num_samples=5000, sampling_method='random'):
    if sampling_method == 'random':
        verts, normals = sample_points_from_meshes(source_mesh, num_samples, return_normals=True)
        source_pc = Pointclouds(points=verts, normals=normals)
    if sampling_method == 'barycentric':
        verts = barycentric_sampling_from_meshes(source_mesh)
        source_pc = Pointclouds(points=verts)

    loss = point_mesh_face_distance(target_mesh, source_pc)

    return loss