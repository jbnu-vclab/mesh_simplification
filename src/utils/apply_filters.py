import pymeshlab
import os
from pathlib import PurePath
from tqdm import tqdm
from glob import glob
from pytorch3d.io import save_obj, load_obj, load_ply, save_ply
from src.structure.mesh import *

def apply_normalization_and_save(device, file_path: str, out_path: str):
    mesh = load_mesh(device, file_path, normalize=True)
    final_verts, final_faces = mesh.get_mesh_verts_faces(0)
    save_obj(out_path, final_verts, final_faces)

def apply_qem_and_save(file_path: str, out_path: str, target_percentage: float):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    ms.simplification_quadric_edge_collapse_decimation(targetperc=target_percentage)
    ms.save_current_mesh(out_path)

def apply_normalization_to_models(device, folder_path: str, out_folder_path:str):
    gl = glob(folder_path + '/*.obj')

    if not os.path.exists(out_folder_path):
        os.mkdir(out_folder_path)

    for file_path in tqdm(gl):
        file_name = PurePath(file_path).name
        out_path = os.path.join(out_folder_path, file_name)
        apply_normalization_and_save(device, file_path, out_path)

def apply_qem_to_models(folder_path: str, out_folder_path:str, target_percentage: float):
    gl = glob(folder_path + '/*.obj')

    if not os.path.exists(out_folder_path):
        os.mkdir(out_folder_path)

    for file_path in tqdm(gl):
        file_name = PurePath(file_path).name
        out_path = os.path.join(out_folder_path, file_name)
        apply_qem_and_save(file_path, out_path, target_percentage)

    
target_percentage = 0.1
FOLDER_PATH = r'E:/TOSCA_TEST'
OUT_PATH = f'E:/TOSCA_TEST_{target_percentage}'

if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # apply_normalization_to_models(device, FOLDER_PATH, OUT_PATH)
    # apply_qem_to_models(FOLDER_PATH, OUT_PATH, target_percentage)
    gl = glob('E:/TOSCA_TEST/original/*.obj')

    result = ''
    for f in gl:
        filename = f.split('/')[-1].split('.obj')[0]
        filename = filename.split('original\\')[1]
        cur = f'- {filename}\n'
        result += cur
    
    print(result)
