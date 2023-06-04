# Util function for loading meshes
import torch
import trimesh
import os
import numpy as np
from glob import glob
from pytorch3d.io import save_obj, load_obj
from pytorch3d.renderer import *
from pytorch3d.structures import Meshes
from pytorch3d.ops import packed_to_padded
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.io import IO

def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    return mesh

def load_mesh(device, obj_path, normalize=True):
    verts, faces_idx, _ = load_obj(obj_path)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    # textures = TexturesUV(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    if normalize == True:
        mesh = normalize_mesh(mesh)

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

    mesh = convert_textureless_mesh_into_textue_mesh(device, mesh)

    return mesh

def edges_packed_to_faces_idx(mesh: Meshes):
    faces_packed = mesh.faces_packed()
    edges_packed = mesh.edges_packed()
    edges_packed_to_faces_idx = []
    for face_idx, face in enumerate(faces_packed):
        for i in range(3):
            vert1 = face[i]
            vert2 = face[(i + 1) % 3]
            edge_indices = (edges_packed == torch.tensor([vert1, vert2]).to(edges_packed.device)).all(dim=-1).nonzero(as_tuple=False)
            if edge_indices.numel() > 0:
                edge_idx = edge_indices[0].item()
                edges_packed_to_faces_idx.append((edge_idx, face_idx))
    edges_packed_to_faces_idx.sort(key=lambda x: x[0])
    result = []
    current_edge_idx = edges_packed_to_faces_idx[0][0]
    current_faces_idx = []
    for edge_idx, face_idx in edges_packed_to_faces_idx:
        if edge_idx == current_edge_idx:
            current_faces_idx.append(face_idx)
        else:
            result.append(current_faces_idx)
            current_edge_idx = edge_idx
            current_faces_idx = [face_idx]
    result.append(current_faces_idx)
    
    max_len = max(map(len, result))
    padded_result = torch.zeros((len(result), max_len), dtype=torch.int64)
    for i, row in enumerate(result):
        padded_result[i, :len(row)] = torch.tensor(row)
    padded_result = padded_result.squeeze()

    return padded_result

def edges_packed_to_faces_packed(mesh: Meshes):
    faces = mesh.faces_packed_to_edges_packed()
    num_edges = mesh.edges_packed().shape[0]
    edges_to_faces_list = [[] for i in range(num_edges)]
    
    for i, f in enumerate(faces):
        e1, e2, e3 = f[0], f[1], f[2]
        edges_to_faces_list[e1].append(i)
        edges_to_faces_list[e2].append(i)
        edges_to_faces_list[e3].append(i)

    # edges_packed = torch.tensor(edges_to_faces_list, dtype=torch.long)
    edges_packed = edges_to_faces_list
    return edges_packed

def get_sharp_verts(device: torch.device, mesh: Meshes, thr: float):
    thr = thr * (torch.pi / 180)

    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    edges_to_faces = edges_packed_to_faces_packed(mesh)
    face_normals = mesh.faces_normals_packed()

    mask = torch.zeros_like(verts, dtype=torch.bool)
    
    num_sharp_edges = 0
    for i, edge in enumerate(edges_to_faces):
        if len(edge) != 2:
            for v in edges[i]:
                mask[v] = True
            continue

        f1, f2 = edge[0], edge[1]
        f1_normal, f2_normal = face_normals[f1], face_normals[f2]
        dihedral_angle = torch.acos((f1_normal * f2_normal).sum(dim=0))
        
        if dihedral_angle > thr:
            num_sharp_edges += 1
            for v in edges[i]:
                mask[v] = True
            
    return mask

    # edges = mesh.edges_packed()

    # for edge in edges:
        # pass

    # Get the edge-to-face adjacency matrix
    E2F = edges_packed_to_faces_idx(mesh)

    # Get the face normals
    face_normals = mesh.faces_normals_packed().cpu()

    # Compute the dihedral angles between adjacent faces
    max_faces = mesh.num_faces_per_mesh().max().item()
    face_normals_padded = packed_to_padded(face_normals, E2F, max_faces)
    dihedral_angles = torch.acos((face_normals_padded[:, 0] * face_normals_padded[:, 1]).sum(dim=-1))

    # for i, edge in enumerate(edges):
    #     v1, v2 = edge
    #     E2F[i]


    # thr1 = torch.pi - thr / 2.0
    # thr2 = torch.pi + thr / 2.0

    # Find the edges with dihedral angles greater than the threshold
    # edge_mask = (dihedral_angles < thr1) & (dihedral_angles > thr2)
    # edge_mask = torch.where((dihedral_angles < thr1) | (dihedral_angles > thr2), True, False)
    edge_mask = torch.where(dihedral_angles > thr, True, False)
    edges = mesh.edges_packed()[edge_mask]

    # Get the vertices shared by these edges
    vertices = torch.unique(edges)

    sharp_verts = vertices.to(device)

    # mask = torch.zeros_like(mesh.verts_packed(), dtype=torch.bool)
    # mask[sharp_verts] = True
    print(sharp_verts.shape)

    return sharp_verts

def detach_rows(device, verts_shape, indices):
    verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    indices = torch.tensor(indices, dtype=torch.long)
    verts[indices] = verts[indices].detach()  # detach the selected rows
    return verts

# def detach_sharp_verts(deform_verts: torch.Tensor, sharp_verts: torch.Tensor):
#     mask = torch.zeros_like(deform_verts, dtype=torch.bool)
#     mask[sharp_verts] = True
#     detached_offset = torch.where(mask, deform_verts.detach(), deform_verts)
#     # detached_offset = deform_verts.detach().where(mask, deform_verts)
#     return detached_offset

def sharp_verts_mask(deform_verts: torch.Tensor, sharp_verts: torch.Tensor):
    mask = torch.zeros_like(deform_verts, dtype=torch.bool, requires_grad=False)
    mask[sharp_verts] = True

    return mask

if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = "."
    # obj_filename = os.path.join(DATA_DIR, "data/fandisk.obj")
    SAVE_DIR = "./data/after_normalize/"

    before_normalization = glob('./data/ModelNet/*.obj')

    for filepath in before_normalization:
        filename = os.path.basename(filepath)
        before = load_mesh(device, filepath, normalize=True)
        IO().save_mesh(before, SAVE_DIR + filename)

    # # Load obj file
    # verts, faces_idx, _ = load_obj(obj_filename, device=device)
    # faces = faces_idx.verts_idx

    # verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
    # textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # mesh = Meshes(
    #     verts=[verts.to(device)],
    #     faces=[faces.to(device)],
    #     textures=textures
    # )

    
    # final_obj = mesh_convexhull(device, mesh)
    # ch_mesh = mesh_convexhull(device, mesh, 2)
    # ch_mesh = ch_mesh.scale_verts(5.0)

    # final_obj_path = os.path.join('./final_model.obj')
    # save_obj(final_obj_path, ch_mesh.verts_packed(), ch_mesh.faces_packed())
    # print('dsa')