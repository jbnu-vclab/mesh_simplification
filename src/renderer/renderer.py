import torch

# Data structures and functions for rendering
from pytorch3d.renderer import *
from src.shader.edge_shader import GaussianEdgeShader, SimpleEdgeShader
from src.shader.depth_shader import SoftDepthShader, HardDepthShader

def define_multi_view_cam(device, num_views, distance, znear, zfar, size=1.0, cam_type='FoVOrthographic'):
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)

    if cam_type == 'FoVPerspective':
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar)
    elif cam_type == 'FoVOrthographic':
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, min_x=-size, min_y=-size, max_x=size, max_y=size, znear=znear, zfar=zfar)
    else:
        raise ValueError(f"There is no {cam_type}. Use 'FoVPerspective' or 'FoVOrthographic' instead.")

    return cameras

def define_light(device, pointlight_location):
    lights = PointLights(device=device, location=[pointlight_location])
    return lights

def define_renderer(device, image_size, shader_str, cameras, lights, 
                    blur_radius=0.0, faces_per_pixel=1, gaussian_edge_thr=0.01):
    shader = None
    if shader_str.lower() == "softphong":
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    if shader_str.lower() == "hardphong":
        shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    if shader_str.lower() == "softsilhouette":
        shader = SoftSilhouetteShader()
    if shader_str.lower() == "gaussianedge":
        shader = GaussianEdgeShader(device=device, edge_threshold=gaussian_edge_thr)
    if shader_str.lower() == "simpleedge":
        shader = SimpleEdgeShader(device=device)
    if shader_str.lower() == "softdepth":
        shader = SoftDepthShader(device=device)
    if shader_str.lower() == "harddepth":
        shader = HardDepthShader(device=device)
    if not shader:
        raise ValueError(f"{shader_str} Shader not found!")
    
    
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        bin_size=0,
        faces_per_pixel=faces_per_pixel# if shader_str.lower() == "softphong" else 1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )

    return renderer

def render_imgs(renderer, meshes, cameras, lights, num_views):
    if lights != None:
        images = renderer(meshes, cameras=cameras, lights=lights)
    else:
        images = renderer(meshes, cameras=cameras)

    return images