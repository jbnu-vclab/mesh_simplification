# Util function for loading meshes
import torch
import trimesh
import os
import numpy as np
from pytorch3d.io import save_obj, load_obj
from pytorch3d.renderer import *
from pytorch3d.structures import Meshes
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

def load_mesh(device, obj_path, normalize=True):
    verts, faces_idx, _ = load_obj(obj_path)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    if normalize == True:
        verts = mesh.verts_packed()
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

    return mesh

def convert_textureless_mesh_into_textue_mesh(device, mesh):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    return mesh

def mesh_convexhull(device:torch.device, mesh: Meshes, level=2):
    verts = mesh.verts_packed().cpu()
    faces = mesh.faces_packed().cpu()

    hull = trimesh.Trimesh(verts, faces).convex_hull

    hull_verts = torch.tensor(hull.vertices, dtype=torch.float)
    hull_faces = torch.tensor(hull.faces)

    mesh = Meshes(
        verts=[hull_verts.to(device)],
        faces=[hull_faces.to(device)]
    )
    subdivide = SubdivideMeshes()

    for i in range(level):
        mesh = subdivide(mesh)
        # verts = mesh.verts_list()[0]
        # # verts /= verts.norm(p=2, dim=1, keepdim=True)
        # faces = mesh.faces_list()[0]

    mesh = convert_textureless_mesh_into_textue_mesh(device, mesh)

    return mesh

if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = "."
    obj_filename = os.path.join(DATA_DIR, "data/fandisk.obj")

    # Load obj file
    verts, faces_idx, _ = load_obj(obj_filename, device=device)
    faces = faces_idx.verts_idx

    verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    # final_obj = mesh_convexhull(device, mesh)
    ch_mesh = mesh_convexhull(device, mesh, 2)
    ch_mesh = ch_mesh.scale_verts(5.0)

    final_obj_path = os.path.join('./final_model.obj')
    save_obj(final_obj_path, ch_mesh.verts_packed(), ch_mesh.faces_packed())
    print('dsa')