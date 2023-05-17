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
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

import os
import wandb
import numpy as np
from tqdm import tqdm

from src.loss.mesh_chamfer_distance import mesh_chamfer_distance, mesh_distance
from src.structure.mesh import *
from src.utils.convert_PIL_grid_img import convert_PIL_grid_img
from src.renderer.renderer import *
from src.loss.loss import mse_loss, loss_with_random_permutation
# from src.renderer.renderer import define_renderer, define_light, define_multi_view_cam, render_imgs

from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def update_mesh_shape_prior_losses(mesh, loss):
    loss["edge"] = mesh_edge_loss(mesh)
    loss["normal"] = mesh_normal_consistency(mesh)
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

def prepare_renderers(args, meshes, cameras, lights, num_views):
    blur_radius = np.log(1. / 1e-4 - 1.) * args['sigma']

    renderers = {}                                                              
    target_imgs = {}

    renderers['softphong'] = define_renderer(device=device,
                                         image_size=args['image_resolution'],
                                         blur_radius=blur_radius,
                                         faces_per_pixel=args['faces_per_pixel'],
                                         shader_str="SoftPhong",
                                         cameras=cameras,
                                         lights=lights
                                         )
    renderers['silhouette'] = define_renderer(device=device,
                                        image_size=args['image_resolution'],
                                        blur_radius=blur_radius,
                                        faces_per_pixel=args['faces_per_pixel'],
                                        shader_str="SoftSilhouette",
                                        cameras=cameras,
                                        lights=lights
                                    )
    renderers['model_edge'] = define_renderer(device=device,
                                    image_size=args['image_resolution'],
                                    shader_str=args['model_edge_type'],
                                    cameras=cameras,
                                    lights=lights,
                                    gaussian_edge_thr=args['gaussian_edge_thr']
                                )
    renderers['depth'] = define_renderer(device=device,
                                    image_size=args['image_resolution'],
                                    blur_radius=blur_radius,
                                    faces_per_pixel=args['faces_per_pixel'],
                                    shader_str="SoftDepth",
                                    cameras=cameras,
                                    lights=None,
                                )
    target_imgs['softphong'] = render_imgs(renderers['softphong'], meshes, cameras, lights, num_views)
    softphong_imgs_grid = convert_PIL_grid_img(target_imgs['softphong'], target_channel=None, nrow=5)

    target_imgs['silhouette'] = render_imgs(renderers['silhouette'], meshes, cameras, lights, num_views)
    silhouette_imgs_grid = convert_PIL_grid_img(target_imgs['silhouette'], target_channel=3, nrow=5)

    target_imgs['model_edge'] = render_imgs(renderers['model_edge'], meshes, cameras, lights, num_views)
    edge_imgs_grid = convert_PIL_grid_img(target_imgs['model_edge'], target_channel=0, nrow=5)

    target_imgs['depth'] = render_imgs(renderers['depth'], meshes, cameras, lights=None, num_views=num_views)
    depth_imgs_grid = convert_PIL_grid_img(target_imgs['depth'], target_channel=0, nrow=5)

    wandb.log({
        "GT Model Img": softphong_imgs_grid,
        "GT Silhouette Img": silhouette_imgs_grid,
        "GT Edge Img": edge_imgs_grid,
        "GT Depth Img": depth_imgs_grid
    })

    return renderers, target_imgs

def prepare_GT(args, obj_path, num_views):
    target_mesh = load_mesh(device, obj_path, normalize=True)
    meshes = target_mesh.extend(num_views)

    cameras = define_multi_view_cam(device=device,
                                    num_views=num_views,
                                    distance=args['cam_distance'],
                                    znear=args['znear'],
                                    zfar=args['zfar'],
                                    size=args['orthographic_size'],
                                    cam_type=args['cam_type']
                                    )

    lights = define_light(device=device,
                          pointlight_location=[0.,0.,-3.])
    
    renderers, target_imgs = prepare_renderers(args, meshes, cameras, lights, num_views)

    return cameras, lights, target_mesh, renderers, target_imgs

def train_test(args):
    DATA_DIR = "data"
    obj_path = os.path.join(DATA_DIR, f"{args['objfile']}.obj")
    num_views = args['num_views']
    blur_radius = np.log(1. / 1e-4 - 1.) * args['sigma']

    cameras, lights, target_mesh, renderers, target_imgs = prepare_GT(args, obj_path, num_views)

    if args['init_src_mesh_type'] == 'ico_sphere':
        src_mesh = ico_sphere(int(args['init_sphere_level']), device)
    if args['init_src_mesh_type'] == 'simplified':
        src_obj_path = os.path.join(DATA_DIR, f"{args['objfile']}_simplified.obj")
        src_mesh = load_mesh(device, src_obj_path, normalize=True)
        subdivide = SubdivideMeshes()
        src_mesh = subdivide(src_mesh)
    if args['init_src_mesh_type'] == 'convexhull':
        origin_obj_path = os.path.join(DATA_DIR, f"{args['objfile']}.obj")
        origin_mesh = load_mesh(device, origin_obj_path, normalize=True)
        src_mesh = mesh_convexhull(device, origin_mesh, args['convexhull_subdiv_level'])
    
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    if (args['init_src_mesh_type'] == 'simplified') & (args['fixed_sharp_verts'] == True):
        sharp_verts = get_sharp_verts(device, src_mesh, args['sharpness_threshold'])
        mask = sharp_verts_mask(deform_verts, sharp_verts)
    else:
        src_mesh = src_mesh.scale_verts(args['init_src_mesh_scale'])
        mask = torch.zeros_like(deform_verts, dtype=torch.bool)

    num_views_per_iteration = args['num_views_per_iteration']
    iter = args['iter']

    losses = {
        "silhouette": {"weight": args['loss_silhouette_weight'], "values": []},
        "depth": {"weight": args['loss_depth_weight'], "values": []},
        "edge": {"weight": args['loss_edge_weight'], "values": []},
        "normal": {"weight": args['loss_normal_weight'], "values": []},
        "laplacian": {"weight": args['loss_laplacian_weight'], "values": []},
        "model_edge": {"weight": args['loss_model_edge_weight'], "values": []},
        "chamfer_distance": {"weight": args['loss_chamfer_distance_weight'], "values": []},
    }

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=args['lr'], momentum=args['momentum'])

    loop = tqdm(range(iter))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts * ~mask)

        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)

        if args['use_silhouette_loss']:
            loss['silhouette'] = loss_with_random_permutation(
                num_views, num_views_per_iteration, renderers['silhouette'], new_src_mesh, cameras, lights, 
                target_imgs['silhouette'], loss_func=mse_loss, target_channel=3)
        
        if args['use_depth_loss']:
            loss['depth'] = loss_with_random_permutation(
                num_views, num_views_per_iteration, renderers['depth'], new_src_mesh, cameras, lights, 
                target_imgs['depth'], loss_func=mse_loss, target_channel=0)

        if args['use_model_edge_loss']:
            loss['model_edge'] = loss_with_random_permutation(
                num_views, num_views_per_iteration, renderers['model_edge'], new_src_mesh, cameras, lights, 
                target_imgs['model_edge'], loss_func=mse_loss, target_channel=0)

        if args['use_cd_loss']:
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

        target_imgs['softphong'] = render_imgs(renderers['softphong'], predicted_mesh, cameras, lights, num_views)
        softphong_imgs_grid = convert_PIL_grid_img(target_imgs['softphong'], target_channel=None, nrow=5)

        target_imgs['silhouette'] = render_imgs(renderers['silhouette'], predicted_mesh, cameras, lights, num_views)
        silhouette_imgs_grid = convert_PIL_grid_img(target_imgs['silhouette'], target_channel=3, nrow=5)

        target_imgs['model_edge'] = render_imgs(renderers['model_edge'], predicted_mesh, cameras, lights, num_views)
        edge_imgs_grid = convert_PIL_grid_img(target_imgs['model_edge'], target_channel=0, nrow=5)

        target_imgs['depth'] = render_imgs(renderers['depth'], predicted_mesh, cameras, lights=None, num_views=num_views)
        depth_imgs_grid = convert_PIL_grid_img(target_imgs['depth'], target_channel=0, nrow=5)

        wandb.log({
            "Test Phong Img": softphong_imgs_grid,
            "Test Silhouette Img": silhouette_imgs_grid,
            "Test Edge Img": edge_imgs_grid,
            "Test Depth Img": depth_imgs_grid
        })

    result_table = wandb.Table(columns=["CD", "Dist result pc -> target mesh", "Dist target pc -> result mesh"])

    final_CD = mesh_chamfer_distance(new_src_mesh, target_mesh, args['chamfer_distance_num_samples'])
    final_mesh_dist_forward = mesh_distance(new_src_mesh, target_mesh, args['mesh_distance_num_samples'])
    final_mesh_dist_backward = mesh_distance(target_mesh, new_src_mesh, args['mesh_distance_num_samples'])
    result_table.add_data(final_CD, final_mesh_dist_forward, final_mesh_dist_backward)
    
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    final_obj = os.path.join(wandb.run.dir, 'final_model.obj')
    save_obj(final_obj, final_verts, final_faces)

    wandb.log({
        "Final Model": wandb.Object3D(open(final_obj)),
        "Result": result_table
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
    args = {}

    #* Wandb
    args['wandb_mode'] = 'online'                 # 'online' for logging, 'disabled' for debug
    # args['wandb_mode'] = 'disabled'              

    #* Data preparation
    args['objfile'] = 'stanford_bunny'
    args['init_sphere_level'] = 3
    args['init_src_mesh_type'] = 'simplified'       # 'ico_sphere' or 'simplified' or 'convexhull'
    args['init_src_mesh_scale'] = 1.2               # scale factor of init source mesh
    args['convexhull_subdiv_level'] = 1             # N of subdivision of convexhull result
    args['fixed_sharp_verts'] = False                # Detach vertices on sharp line
    args['sharpness_threshold'] = 10                # If dihedral angle is higher than threshold, it will be fixed 

    #* Camera
    args['cam_distance'] = 2.5                      # Distance between camera and object
    args['num_views'] = 12
    args['znear'] = 1.0
    args['zfar'] = 4.0
    args['cam_type'] = 'FoVOrthographic'            # 'FoVPerspective' or 'FoVOrthographic'
    args['orthographic_size'] = 1.3                 # size of orthographic camera

    #* Renderer
    args['sigma'] = 1e-5                            # refer SoftRas
    args['faces_per_pixel'] = 10                    # SoftRas
    args['image_resolution'] = 512

    #* Training
    args['num_views_per_iteration'] = 4
    args['iter'] = 500

    args['shaders'] = ['silhouette', 'model_edge', 'depth']
    args['use_silhouette_loss'] = True
    args['use_depth_loss'] = True
    args['use_model_edge_loss'] = True
    args['use_cd_loss'] = True
    args['chamfer_distance_num_samples'] = 40000    # N of samples of mesh when calculate chamfer distance
    args['mesh_distance_num_samples'] = 40000
    args['model_edge_type'] = 'GaussianEdge'        # 'GaussianEdge' or 'SimpleEdge'
    args['gaussian_edge_thr'] = 0.01

    # loss weights
    args['loss_edge_weight'] = 0.2
    args['loss_laplacian_weight'] = 0.02
    args['loss_normal_weight'] = 0.01
    args['loss_silhouette_weight'] = 1.0
    args['loss_model_edge_weight'] = 1.0
    args['loss_depth_weight'] = 1.0
    args['loss_chamfer_distance_weight'] = 1.0      

    args['lr'] = 0.3
    args['momentum'] = 0.9

    wandb.init(entity='jbnu-vclab', project="mesh_simplification", reinit=True, mode=args['wandb_mode'])
    wandb.config.update(args)

    train_test(args)

    wandb.finish(quiet=True) # 'running' status forever w/o this line