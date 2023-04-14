import torch

# Data structures and functions for rendering
from pytorch3d.renderer import *
from ..shader.edge_shader import GaussianEdgeShader, SimpleEdgeShader

def define_multi_view_cam(device, num_views, distance):
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    return cameras

def define_light(device, pointlight_location):
    lights = PointLights(device=device, location=[pointlight_location])
    return lights

def define_renderer(device, image_size, blur_radius, faces_per_pixel,
                    shader_str, cameras, lights, args, plot=False):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
    )

    shader = None
    if shader_str.lower() == "softphong":
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    if shader_str.lower() == "softsilhouette":
        shader = SoftSilhouetteShader()
    if shader_str.lower() == "gaussianedge":
        shader = GaussianEdgeShader(device=device, edge_threshold=args['gaussian_edge_thr'])
    if shader_str.lower() == "simpleedge":
        shader = SimpleEdgeShader(device=device)
    if not shader:
        raise ValueError(f"{shader_str} Shader not found!")

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )

    return renderer

def render_imgs(renderer, meshes, cameras, lights, num_views):
    images = renderer(meshes, cameras=cameras, lights=lights)

    return images