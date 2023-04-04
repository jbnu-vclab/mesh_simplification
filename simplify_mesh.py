import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm, trange

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import *

from my_shader import ModelEdgeShader, SimpleEdgeShader

import os

from utils.plot_image_grid import image_grid

import wandb

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def load_mesh(obj_path, normalize=True):
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

def define_multi_view_cam(num_views, distance):
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    return cameras

def define_light(pointlight_location):
    lights = PointLights(device=device, location=[pointlight_location])
    return lights

def define_renderer(image_size, blur_radius, faces_per_pixel,
                    shader_str, cameras, lights):
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
    if shader_str.lower() == "edge":
        shader = ModelEdgeShader(device=device)
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

# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.show()

def update_mesh_shape_prior_losses(mesh, loss):
    loss["edge"] = mesh_edge_loss(mesh)
    loss["normal"] = mesh_normal_consistency(mesh)
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


def prepare_GT(args, obj_path, num_views):
    mesh = load_mesh(obj_path, normalize=True)
    meshes = mesh.extend(num_views)

    cameras = define_multi_view_cam(num_views=num_views, distance=args['cam_distance'])

    lights = define_light([0.,0.,-3.])

    softphong_renderer = define_renderer(image_size=args['image_resolution'],
                                           blur_radius=0.0,
                                           faces_per_pixel=1,
                                            shader_str="SoftPhong",
                                         cameras=cameras,
                                         lights=lights)

    softphong_imgs = render_imgs(softphong_renderer, meshes, cameras, lights, num_views)
    fig_gt_phong = image_grid(softphong_imgs.cpu().numpy(), rows=2, cols=5, rgb=True)

    silhouette_renderer = define_renderer(args['image_resolution'], args['blur_radius'], 50, "SoftSilhouette", cameras, lights)
    target_silhouette_imgs = render_imgs(silhouette_renderer, meshes, cameras, lights, num_views)
    fig_gt_sil = image_grid(softphong_imgs.cpu().numpy(), rows=2, cols=5, rgb=False)

    edge_renderer = define_renderer(args['image_resolution'], 0.0, 1, "SimpleEdge", cameras, lights)
    target_edge_imgs = render_imgs(edge_renderer, meshes, cameras, lights, num_views)
    fig_gt_edge = image_grid(softphong_imgs.cpu().numpy(), rows=2, cols=5, rgb=True)

    wandb.log({
        "GT Model Img": fig_gt_phong,
        "GT Silhouette Img": fig_gt_sil,
        "GT Edge Img": fig_gt_edge
    })

    return cameras, lights, target_silhouette_imgs, target_edge_imgs

def train_test(args):
    DATA_DIR = "./data"
    obj_path = os.path.join(DATA_DIR, f"{args['objfile']}.obj")
    num_views = args['num_views']

    cameras, lights, target_silhouette_imgs, target_edge_imgs = prepare_GT(args,obj_path, num_views)

    src_mesh = ico_sphere(args['init_sphere_level'], device)
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    silhouette_renderer = define_renderer(args['image_resolution'], args['blur_radius'], 50, "SoftSilhouette", cameras, lights)
    edge_renderer = define_renderer(args['image_resolution'], 0.0, 1, "SimpleEdge", cameras, lights)



    num_views_per_iteration = args['num_views_per_iteration']
    iter = args['iter']

    losses = {
        "silhouette": {"weight": args['loss_silhouette_weight'], "values": []},
        "edge": {"weight": args['loss_edge_weight'], "values": []},
        "normal": {"weight": args['loss_normal_weight'], "values": []},
        "laplacian": {"weight": args['loss_laplacian_weight'], "values": []},
        "model_edge": {"weight": args['loss_model_edge_weight'], "values": []},
    }

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=args['lr'], momentum=args['momentum'])

    loop = tqdm(range(iter))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)

        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)

        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = silhouette_renderer(new_src_mesh, cameras=cameras[j], lights=lights)
            predicted_silhouette = images_predicted[..., 3]
            target_silhouette = target_silhouette_imgs[j, ..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette) ** 2).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration

        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = edge_renderer(new_src_mesh, cameras=cameras[j], lights=lights)
            predicted_model_edge = images_predicted[..., 0]
            target_edge = target_edge_imgs[j, ..., 0]
            loss_model_edge = ((predicted_model_edge - target_edge) ** 2).mean()
            loss["model_edge"] += loss_model_edge / num_views_per_iteration

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))

            wandb.log({
                k: l
            })

        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)

        # Optimization step음
        sum_loss.backward()
        optimizer.step()

    # Plot mesh
    with torch.no_grad():
        predicted_mesh = new_src_mesh.detach().extend(num_views)
        predicted_silhouette = silhouette_renderer(predicted_mesh, cameras=cameras, lights=lights)
        fig_test_sil = image_grid(predicted_silhouette.cpu().numpy(), rows=2, cols=5, rgb=True)

        predicted_edge = edge_renderer(predicted_mesh, cameras=cameras, lights=lights)
        fig_test_edge = image_grid(predicted_edge.cpu().numpy(), rows=2, cols=5, rgb=True)

        # TODO: plt 저장 잘 안되는 경우가 있음
        wandb.log({
            "Test Silhouette Img": fig_test_sil,
            "Test Edge Img": fig_test_edge
        })


    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    final_obj = os.path.join('./', 'final_model_w_edge_loss.obj')
    save_obj(final_obj, final_verts, final_faces)
    # TODO: WandB File Loggning

    plot_losses(losses)

    # TODO
    # 1) 처음에 실루엣 기반으로 전체 모양을 잡고, edge로 파인 튜닝을 하는 개념으로 접근?
    #    - 어쨋든 model edge는 관여하는 픽셀 개수 차이가 매우 크기 때문에 weight가 아주 커야
    # 2) 삼각형을 어느 정도로 세분화 해서 시작해야 하나? 패러럴하게 하거나 offset을 다른 상세도의 모델로 전파하는 방법도 생각해 봐야
    # 3) CNN을 융합하는 것은 계속 고려
    # 4) CD loss를 추가적으로 활용하는 방법?
    # 5) ailiasing으로 인해 낮은 해상도에서는 의도치 않은 효과 발생 가능
    # 6) 타겟 이미지 생성시에는 skeleton화를 하는 것은 어떨지?

if __name__ == "__main__":
    wandb.init(entity='jbnu-vclab', project="mesh_simplification", reinit=True)
    # wandb.run.name = 'your-run-name'
    # wandb.run.save()

    args = {}
    args['objfile'] = 'fandisk'
    args['num_views'] = 10
    args['cam_distance'] = 2.7
    args['sigma'] = 1e-4
    args['blur_radius'] = np.log(1. / 1e-4 - 1.) * args['sigma']
    args['image_resolution'] = 512
    args['init_sphere_level'] = 4
    args['num_views_per_iteration'] = 4
    args['iter'] = 500
    args['loss_silhouette_weight'] = 0.9
    args['loss_edge_weight'] = 0.8
    args['loss_normal_weight'] = 0.01
    args['loss_laplacian_weight'] = 1.0
    args['loss_model_edge_weight'] = 1.0
    args['lr'] = 1.0
    args['momentum'] = 0.9

    wandb.config.update(args)

    train_test(args)

