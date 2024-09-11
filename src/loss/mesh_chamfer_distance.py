import torch
import numpy as np
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from src.ops.sample_points_from_meshes import barycentric_sampling_from_meshes

def mesh_chamfer_distance(source_mesh, target_mesh, num_samples=5000, sampling_method='random', norm=2):
    if sampling_method == 'random':
        source_points, source_normals = sample_points_from_meshes(source_mesh, num_samples, return_normals=True)
        target_points, target_normals = sample_points_from_meshes(target_mesh, num_samples, return_normals=True)
    elif sampling_method == 'barycentric':
        source_points, source_normals = barycentric_sampling_from_meshes(source_mesh)
        # print(source_points.shape)
        # print(source_normals.shape)
        target_points, target_normals = barycentric_sampling_from_meshes(target_mesh)
        # target_points, target_normals = sample_points_from_meshes(target_mesh, source_points.shape[1], return_normals=True)
        # print(target_points.shape)
        # print(target_normals.shape)

    spcl = Pointclouds(source_points, source_normals)
    tpcl = Pointclouds(target_points, target_normals)

    loss, _ = chamfer_distance(x=source_points, 
                               y=target_points, 
                               x_normals=source_normals, 
                               y_normals=target_normals,
                               norm=norm)

    return loss, spcl, tpcl

def mesh_hausdorff_distance(source_mesh, target_mesh, num_samples=5000, sampling_method='random'):
    if sampling_method == 'random':
        source_points, source_normals = sample_points_from_meshes(source_mesh, num_samples, return_normals=True)
        target_points, target_normals = sample_points_from_meshes(target_mesh, num_samples, return_normals=True)
    elif sampling_method == 'barycentric':
        source_points, source_normals = barycentric_sampling_from_meshes(source_mesh)
        target_points, target_normals = barycentric_sampling_from_meshes(target_mesh)

    spcl = Pointclouds(source_points, source_normals)
    tpcl = Pointclouds(target_points, target_normals)

    loss = chamfer_distance(x=source_points,
                            y=target_points,
                            x_normals=source_normals,
                            y_normals=target_normals,
                            point_reduction='max')

    return loss, spcl, tpcl

def point_to_mesh_distance(source_mesh, target_mesh, num_samples=5000, sampling_method='random'):
    if sampling_method == 'random':
        verts, normals = sample_points_from_meshes(source_mesh, num_samples, return_normals=True)
        source_pc = Pointclouds(points=verts, normals=normals)
    if sampling_method == 'barycentric':
        verts, normals = barycentric_sampling_from_meshes(source_mesh)
        # source_pc = Pointclouds(points=verts, normals=normals)
        source_pc = Pointclouds(points=verts)

    loss = point_mesh_face_distance(target_mesh, source_pc)

    return loss