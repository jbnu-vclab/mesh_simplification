# Util function for point clouds
import torch
import trimesh
import os
import numpy as np
from pytorch3d.io import save_obj, load_obj
from pytorch3d.renderer import *
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import packed_to_padded
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

def verts_to_pointclouds(mesh: Meshes):
    verts = mesh.verts_packed()
    pcl = Pointclouds(points=verts)
    return pcl