import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

# Setup
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
#
# Normalize mesh
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))

lights = PointLights(device=device, location=[[0.0, 0.0, -4.0]])

R, T = look_at_view_transform(dist=3, elev=0, azim=30, at=((0, -0.25, 0),), up=((0,1,0),))
# cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.001, zfar=3)
cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=2, zfar=4)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer_phong = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)
#
# phong_img = renderer_phong(mesh, cameras=cameras, lights=lights)
# plt.figure(figsize=(10, 10))
# plt.imshow(phong_img[0, ..., :3].cpu().numpy())
# plt.axis("off")
# plt.show()

from src.shader.edge_shader import GaussianEdgeShader

raster_settings_edge = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer_edge = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_edge
    ),
    shader=GaussianEdgeShader(
        device=device,
        edge_threshold=0.012
    )
)

edge_img = renderer_edge(mesh, cameras=cameras, lights=lights)


plt.figure(figsize=(10, 10))
plt.imshow(edge_img[0, ..., 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
plt.axis("off")
plt.show()

# rasterizer = MeshRasterizer(
#     cameras=cameras,
#     raster_settings=raster_settings
# )
#
# fragments = rasterizer(mesh)
#
# fragments_zbuf = fragments.zbuf[0, ..., 0]
# mask = fragments_zbuf < 3
#
#
# depth_map = fragments_zbuf / 3
#
# plt.figure(figsize=(10, 10))
# plt.imshow(depth_map.cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
# plt.axis("off")
# plt.show()
