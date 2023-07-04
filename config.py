args = {
    #* Logger
    'wandb_entity': 'jbnu-vclab',
    'wandb_project': 'mesh_simplification',
    'wandb_reinit': True,
    'wandb_mode': 'online',                    # 'online' for logging, 'disabled' for debug

    #* Data
    'objfile' : 'cat124',
    'result_path' : 'E:/Results/',
    'init_sphere_level' : 4,
    'init_src_mesh_type' : 'simplified',       # 'ico_sphere' or 'simplified' or 'convexhull'
    'simplify_level' : 0.05,
    'init_src_mesh_scale' : 1.0,               # scale factor of init source mesh
    'convexhull_subdiv_level' : 1,             # N of subdivision of convexhull result
    'fixed_sharp_verts' : False,               # Detach vertices on sharp line
    'sharpness_threshold' : 40,                # If dihedral angle is higher than threshold, it will be fixed 
    'normalize_source_mesh' : False,
    'normalize_target_mesh' : False,

    #* Renderer
    'cam_distance' : 1.8,                      # Distance between camera and object
    'num_views' : 4,
    'znear' : 0.5,
    'zfar' : 3.0,
    'cam_type' : 'FoVOrthographic',            # 'FoVPerspective' or 'FoVOrthographic'
    'orthographic_size' : 1.5,                 # size of orthographic camera
    'sigma' : 1e-5,                            # refer SoftRas
    'faces_per_pixel' : 10,                    # SoftRas
    'image_resolution' : 800,
    'model_edge_type' : 'GaussianEdge',        # 'GaussianEdge' or 'SimpleEdge'
    'gaussian_edge_thr' : 0.01,

    #* Training
    'num_views_per_iteration' : 4,
    'iter' : 1000,
    'lr' : 0.1,
    'momentum' : 0.9,
    'cd_num_samples' : 50000,                   # N of samples of mesh when calculate CD
    'metric_num_samples' : 50000,               # N of samples of mesh when calculate metric (CD and point-to-face dist) 
    'edge_target_length' : 0.0,                 # target length of edge loss (notice: not model edge loss)
    'laplacian_method' : 'uniform',             # 'uniform' or 'cot' or 'cotcurv' (for laplacian smoothing)
    'cd_sampling_method' : 'random',          # 'random' or 'barycentric'
    'md_sampling_method' : 'barycentric',          # 'random' or 'barycentric'
    'silhouette_loss_type' : 'iou',             # silhouette loss calculation type ('mse' or 'iou')

    'use_silhouette_loss' : False,
    'use_depth_loss' : False,
    'use_model_edge_loss' : False,
    'use_cd_loss' : False,
    'use_md_loss' : True,                       # Point(source) to Mesh(target) distance loss

    'loss_edge_weight' : 0.0,
    'loss_laplacian_weight' : 0.0,
    'loss_normal_weight' : 0.00,
    'loss_silhouette_weight' : 1.0,
    'loss_depth_weight' : 1.0,
    'loss_model_edge_weight' : 1.0,
    'loss_cd_weight' : 1.0, 
    'loss_md_weight' : 1.0,

    #* Plot
    'plot_images': False,
}