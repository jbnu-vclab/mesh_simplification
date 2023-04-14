import torch

from pytorch3d.utils import ico_sphere
# Util function for loading meshes
from pytorch3d.io import save_obj, load_obj

from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from pytorch3d.renderer import *

import os
import wandb
import numpy as np
from tqdm import tqdm

from src.loss.mesh_chamfer_distance import mesh_chamfer_distance
from src.structure.mesh import *
from src.utils.convert_PIL_grid_img import convert_PIL_grid_img
from src.renderer.renderer import *
# from src.renderer.renderer import define_renderer, define_light, define_multi_view_cam, render_imgs

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def update_mesh_shape_prior_losses(mesh, loss):
    loss["edge"] = mesh_edge_loss(mesh)
    loss["normal"] = mesh_normal_consistency(mesh)
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

def prepare_GT(args, obj_path, num_views):
    target_mesh = load_mesh(device, obj_path, normalize=True)
    meshes = target_mesh.extend(num_views)

    cameras = define_multi_view_cam(device=device,
                                    num_views=num_views,
                                    distance=args['cam_distance'])

    lights = define_light(device=device,
                          pointlight_location=[0.,0.,-3.])

    softphong_renderer = define_renderer(device=device,
                                         image_size=args['image_resolution'],
                                         blur_radius=0.0,
                                         faces_per_pixel=1,
                                         shader_str="SoftPhong",
                                         cameras=cameras,
                                         args=args,
                                         lights=lights
                                         )

    softphong_imgs = render_imgs(softphong_renderer, meshes, cameras, lights, num_views)
    softphong_imgs_grid = convert_PIL_grid_img(softphong_imgs, target_channel=None, nrow=5)

    silhouette_renderer = define_renderer(device, args['image_resolution'], args['blur_radius'], 50, "SoftSilhouette", cameras, lights, args=args)
    target_silhouette_imgs = render_imgs(silhouette_renderer, meshes, cameras, lights, num_views)
    silhouette_imgs_grid = convert_PIL_grid_img(target_silhouette_imgs, target_channel=3, nrow=5)

    edge_renderer = define_renderer(device, args['image_resolution'], 0.0, 1, args['model_edge_type'], cameras, lights, args=args)
    target_edge_imgs = render_imgs(edge_renderer, meshes, cameras, lights, num_views)
    edge_imgs_grid = convert_PIL_grid_img(target_edge_imgs, target_channel=0, nrow=5)


    wandb.log({
        "GT Model Img": softphong_imgs_grid,
        "GT Silhouette Img": silhouette_imgs_grid,
        "GT Edge Img": edge_imgs_grid
    })

    return cameras, lights, target_mesh, target_silhouette_imgs, target_edge_imgs

def train_test(args):
    DATA_DIR = "data"
    obj_path = os.path.join(DATA_DIR, f"{args['objfile']}.obj")
    num_views = args['num_views']

    cameras, lights, target_mesh, target_silhouette_imgs, target_edge_imgs = prepare_GT(args,obj_path, num_views)

    if args['init_src_mesh_type'] == 'ico_sphere':
        src_mesh = ico_sphere(int(args['init_sphere_level']), device)
    elif args['init_src_mesh_type'] == 'simplified':
        src_obj_path = os.path.join(DATA_DIR, f"{args['objfile']}_simplified.obj")
        src_mesh = load_mesh(device, src_obj_path, normalize=True)

    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    silhouette_renderer = define_renderer(device=device,
                                        image_size=args['image_resolution'],
                                        blur_radius=args['blur_radius'],
                                        faces_per_pixel=50,
                                        shader_str="SoftSilhouette",
                                        cameras=cameras,
                                        lights=lights,
                                        args=args)

    edge_renderer = define_renderer(device=device,
                                    image_size=args['image_resolution'],
                                    blur_radius=0.0,
                                    faces_per_pixel=1,
                                    shader_str=args['model_edge_type'],
                                    cameras=cameras,
                                    lights=lights,
                                    args=args)

    softphong_renderer = define_renderer(device=device,
                                        image_size=args['image_resolution'],
                                        blur_radius=0.0,
                                        faces_per_pixel=1,
                                        shader_str="SoftPhong",
                                        cameras=cameras,
                                        lights=lights,
                                        args=args)

    num_views_per_iteration = args['num_views_per_iteration']
    iter = args['iter']

    losses = {
        "silhouette": {"weight": args['loss_silhouette_weight'], "values": []},
        "edge": {"weight": args['loss_edge_weight'], "values": []},
        "normal": {"weight": args['loss_normal_weight'], "values": []},
        "laplacian": {"weight": args['loss_laplacian_weight'], "values": []},
        "model_edge": {"weight": args['loss_model_edge_weight'], "values": []},
        "chamfer_distance": {"weight": args['loss_chamfer_distance_weight'], "values": []},
    }
    #TODO: Depth loss 추가

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

        if args['use_silhouette']:
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
            # Cross-Entropy
            # cee_loss = torch.nn.CrossEntropyLoss()
            # model_edge_cee = cee_loss(predicted_model_edge, target_edge)
            model_edge_mse = ((predicted_model_edge - target_edge) ** 2).mean()

            loss_gaussian_edge = model_edge_mse

            loss["model_edge"] += loss_gaussian_edge / num_views_per_iteration

        loss["chamfer_distance"] = mesh_chamfer_distance(new_src_mesh, target_mesh, args['chamfer_distance_num_samples'])

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))

            wandb.log({
                k: l
            })

        wandb.log({
            'total_loss': sum_loss,
        })

        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)

        # Optimization step
        sum_loss.backward()
        optimizer.step()


    # Plot mesh
    with torch.no_grad():
        predicted_mesh = new_src_mesh.detach().extend(num_views)

        predicted_mesh = convert_textureless_mesh_into_textue_mesh(device, predicted_mesh)
        predicted_mesh = predicted_mesh.extend(args["num_views"])

        predicted_silhouette = silhouette_renderer(predicted_mesh, cameras=cameras, lights=lights)
        silhouette_imgs_grid = convert_PIL_grid_img(predicted_silhouette, target_channel=3, nrow=5)

        predicted_edge = edge_renderer(predicted_mesh, cameras=cameras, lights=lights)
        edge_imgs_grid = convert_PIL_grid_img(predicted_edge, target_channel=0, nrow=5)

        predicted_phong = softphong_renderer(predicted_mesh, cameras=cameras, lights=lights)
        phong_imgs_grid = convert_PIL_grid_img(predicted_phong, target_channel=None, nrow=5)

        wandb.log({
            "Test Silhouette Img": silhouette_imgs_grid,
            "Test Edge Img": edge_imgs_grid,
            "Test Phong Img": phong_imgs_grid,
        })

    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    final_obj = os.path.join(wandb.run.dir, 'final_model.obj')
    save_obj(final_obj, final_verts, final_faces)

    #! 오브젝트 logging이 됐다 안됐다 함
    wandb.log({
        "Final Model": wandb.Object3D(open(final_obj))
    })

    wandb.save(final_obj)


    # TODO:
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
    args['init_sphere_level'] = 3
    args['num_views_per_iteration'] = 4
    args['iter'] = 500
    args['use_silhouette'] = True
    args['loss_silhouette_weight'] = 1.0
    args['loss_edge_weight'] = 0.8
    args['loss_normal_weight'] = 0.01
    args['loss_laplacian_weight'] = 0.0
    args['loss_model_edge_weight'] = 1.0
    args['loss_chamfer_distance_weight'] = 1.0
    args['chamfer_distance_num_samples'] = 10000
    args['model_edge_type'] = 'GaussianEdge' # 'GaussianEdge' or 'SimpleEdge'
    args['init_src_mesh_type'] = 'ico_sphere' # 'ico_sphere' or 'simplified'
    args['gaussian_edge_thr'] = 0.012
    args['lr'] = 1.0
    args['momentum'] = 0.9

    wandb.config.update(args)

    train_test(args)

    wandb.finish(quiet=True) # remains 'running' status w/o this line