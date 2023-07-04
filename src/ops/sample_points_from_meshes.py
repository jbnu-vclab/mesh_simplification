import sys
from typing import Tuple, Union

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded

def barycentric_sampling_from_meshes(
    meshes
) -> torch.Tensor:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.

    Returns:
        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    num_meshes = len(meshes)

    #! We assume there is only one mesh in meshes
    assert(num_meshes == 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    barycentric_coords = compute_barycentric_coordinates(v0, v1, v2)

    samples = barycentric_coords.unsqueeze(0)

    return samples

def compute_barycentric_coordinates(v0: torch.Tensor, v1:torch.Tensor, v2:torch.Tensor) -> torch.Tensor:
    barycentric_coords = (v0 + v1 + v2) / 3.0
    return barycentric_coords