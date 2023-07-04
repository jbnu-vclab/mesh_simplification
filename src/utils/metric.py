import torch
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.io import save_obj, load_obj, load_ply, save_ply
from src.structure.mesh import *
from src.loss.mesh_chamfer_distance import *
from pytorch3d.io import IO
from pathlib import PurePath
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import pandas as pd

def calc_metric(source_mesh: Meshes, target_mesh: Meshes, num_samples, norm=2, k=5):
    assert(k >= 1)

    metrics = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        cd, spcl, tpcl = mesh_chamfer_distance(source_mesh, target_mesh, num_samples, norm=norm)

        # IO().save_pointcloud(spcl, "./source_pointcloud.ply")
        # IO().save_pointcloud(tpcl, "./target_pointcloud.ply")

        src2gt = point_to_mesh_distance(target_mesh, source_mesh, num_samples)
        gt2src = point_to_mesh_distance(source_mesh, target_mesh, num_samples)

        cd = cd.item()
        src2gt = src2gt.item()
        gt2src = gt2src.item()

        metrics[i] = cd, src2gt, gt2src
    
    cd, src2gt, gt2src = metrics.mean(axis=0)

    return cd, src2gt, gt2src

def calc_average_metrics(source_folder_path: str, target_folder_path: str, norm=2):
    target_models = glob(f'{target_folder_path}/*.obj')
    source_models = glob(f'{source_folder_path}/*.obj')

    assert(len(target_models) == len(source_models))

    metrics = np.zeros((len(target_models), 3), dtype=np.float32)
    names = []
    for i, source_model_path in enumerate(tqdm(source_models)):
        file_name = PurePath(source_model_path).name
        names.append(file_name)
        target_model_path = os.path.join(target_folder_path, file_name)

        if target_model_path in target_models:
            source_mesh = load_mesh(device, source_model_path, normalize=False)
            target_mesh = load_mesh(device, target_model_path, normalize=False)

            # source_mesh = source_mesh.scale_verts(0.5)
            # target_mesh = target_mesh.scale_verts(0.5)

            # cd, src2gt, gt2src = calc_metric(A, B)
            metrics[i] = calc_metric(source_mesh=source_mesh, target_mesh=target_mesh, num_samples=50000, norm=norm)

        else:
            print(f'{file_name} : does not exist in target folder')

    df = pd.DataFrame(names, columns=['name'])
    df['CD'] = metrics[:,0]
    df['src->GT'] = metrics[:, 1]
    df['GT->src'] = metrics[:, 2]

    df.to_csv("./metrics.csv", index = False)

    for i, name in enumerate(names):
        print(f'{name:15}: {round(metrics[i][0], 8):.8f}, {round(metrics[i][1], 8):.8f}, {round(metrics[i][2], 8):.8f}')

    mean = metrics.mean(axis=0)
    print(f'{"--- Average ---" }: {round(mean[0], 8):.8f}, {round(mean[1], 8):.8f}, {round(mean[2], 8):.8f}')


#* 단일 모델 성능 계산 및 점군 생성
if __name__=='__main__':
     
    PATH = './data/'
    file_name = 'cuboid'
    A = load_mesh(device, PATH + file_name + '.obj', normalize=False)
    B = load_mesh(device, PATH + file_name + '_simplified.obj', normalize=False)

    cd, src2gt, gt2src = calc_metric(A, B)

    # IO().save_pointcloud(spcl, "./source_pointcloud.ply")
    # IO().save_pointcloud(tpcl, "./target_pointcloud.ply")

    print(f'{cd}, {src2gt}, {gt2src}')

#* 전체 성능 계산
# if __name__=='__main__':
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#         torch.cuda.set_device(device)
#     else:
#         device = torch.device("cpu")
#     calc_average_metrics('E:/TOSCA_TEST/0.2','E:/TOSCA_TEST/original', norm=2)