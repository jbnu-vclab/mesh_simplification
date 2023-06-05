import torch
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.io import save_obj, load_obj, load_ply, save_ply
from src.structure.mesh import *
from src.loss.mesh_chamfer_distance import *
from pytorch3d.io import IO

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def calc_metric(source_mesh: Meshes, target_mesh: Meshes, num_samples):
    cd, spcl, tpcl = mesh_chamfer_distance(source_mesh, target_mesh, num_samples)

    src2gt = mesh_distance(target_mesh, source_mesh, num_samples)
    gt2src = mesh_distance(source_mesh, target_mesh, num_samples)

    cd = cd.item()
    src2gt = src2gt.item()
    gt2src = gt2src.item()

    return cd, src2gt, gt2src

if __name__=='__main__':
    #TODO: 현재 폴더 구조대로 한번에 모든 Src mesh (Simplified) metric 구해주는 코드 
    PATH = './data/'
    file_name = 'cuboid'
    A = load_mesh(device, PATH + file_name + '.obj', normalize=False)
    B = load_mesh(device, PATH + file_name + '_simplified.obj', normalize=False)

    cd, src2gt, gt2src = calc_metric(A, B)

    # IO().save_pointcloud(spcl, "./source_pointcloud.ply")
    # IO().save_pointcloud(tpcl, "./target_pointcloud.ply")

    print(f'{cd}, {src2gt}, {gt2src}')