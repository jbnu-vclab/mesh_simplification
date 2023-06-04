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

def single_mesh_calc_metric(A_path: str, B_path: str):
    A = load_mesh(device, A_path, normalize=False)
    B = load_mesh(device, B_path, normalize=False)

    cd, spcl, tpcl = mesh_chamfer_distance(A, B, 40000)

    src2gt = mesh_distance(B, A, 40000)
    gt2src = mesh_distance(A, B, 40000)

    return cd, src2gt, gt2src

if __name__=='__main__':
    #TODO: 현재 폴더 구조대로 한번에 모든 Src mesh (Simplified) metric 구해주는 코드 
    PATH = './data/'
    file_name = 'cuboid'
    A = load_mesh(device, PATH + file_name + '.obj', normalize=False)
    B = load_mesh(device, PATH + file_name + '_simplified.obj', normalize=False)

    cd, spcl, tpcl = mesh_chamfer_distance(A, B, 40000)

    # IO().save_pointcloud(spcl, "./source_pointcloud.ply")
    # IO().save_pointcloud(tpcl, "./target_pointcloud.ply")

    print(f'{cd}, {mesh_distance(B, A, 40000)}, {mesh_distance(A, B, 40000)}')