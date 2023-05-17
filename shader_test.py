import os
import torch
import matplotlib.pyplot as plt

from src.renderer.renderer import *
from src.structure.mesh import *

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

num_views = 10

# Set paths
DATA_DIR = "."
obj_filename = os.path.join(DATA_DIR, "data/fandisk.obj")

# Load obj file
target_mesh = load_mesh(device, obj_filename, normalize=True)
meshes = target_mesh.extend(num_views)

cameras = define_multi_view_cam(device=device,
                                num_views=num_views,
                                distance=2.7,
                                znear=1.0,
                                zfar=10.0,
                                size=1.3,
                                cam_type='FoVOrthographic'
                                # cam_type='FoVPerspective'
                                )

lights = define_light(device=device,
                      pointlight_location=[0.,0.,-3.])


blur_radius = np.log(1. / 1e-4 - 1.) * 1e-6
renderer = define_renderer(device=device,
                        image_size=1024,
                        blur_radius=blur_radius,
                        faces_per_pixel=2,
                        shader_str="GaussianEdge",
                        cameras=cameras,
                        lights=lights,
                        )

target_imgs = render_imgs(renderer, meshes, cameras, lights, num_views)

inds = 0
plt.figure(figsize=(20, 100))

for i in range(1, 11):
    plt.subplot(10, 1, i)
    plt.imshow((target_imgs[i-1, ..., 0]*255).cpu().detach().numpy(), cmap='gray')

plt.savefig('./output/shader_test_result.png', bbox_inches='tight', pad_inches=0)

