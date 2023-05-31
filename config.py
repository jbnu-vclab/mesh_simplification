args = {
    #* Logger
    'wandb_entity': 'jbnu-vclab',
    'wandb_project': 'mesh_simplification',
    'wandb_reinit': True,
    'wandb_mode': 'online', # 'online' for logging, 'disabled' for debug

    #* Data
    'objfile' : 'fandisk',
    'init_sphere_level' : 4,
    'init_src_mesh_type' : 'ico_sphere',       # 'ico_sphere' or 'simplified' or 'convexhull'
    'init_src_mesh_scale' : 1.2,               # scale factor of init source mesh
    'convexhull_subdiv_level' : 1,             # N of subdivision of convexhull result
    'fixed_sharp_verts' : True,               # Detach vertices on sharp line
    'sharpness_threshold' : 4,                 # If dihedral angle is higher than threshold, it will be fixed 
    'normalize_source_mesh' : False,
    'normalize_target_mesh' : False,

    #* Renderer
    'cam_distance' : 1.8,                      # Distance between camera and object
    'num_views' : 20,
    'znear' : 0.5,
    'zfar' : 3.0,
    'cam_type' : 'FoVOrthographic',            # 'FoVPerspective' or 'FoVOrthographic'
    'orthographic_size' : 1.5,                 # size of orthographic camera
    'sigma' : 1e-5,                            # refer SoftRas
    'faces_per_pixel' : 10,                    # SoftRas
    'image_resolution' : 512,
    'model_edge_type' : 'GaussianEdge',        # 'GaussianEdge' or 'SimpleEdge'
    'gaussian_edge_thr' : 0.01,

    #* Training
    'num_views_per_iteration' : 2,
    'iter' : 3000,
    'lr' : 1.0,
    'momentum' : 0.9,
    'cd_num_samples' : 40000,                   # N of samples of mesh when calculate chamfer distance
    'mesh_dist_num_samples' : 40000,            # N of samples of mesh when calculate point-to-face dist

    'use_silhouette_loss' : True,
    'use_depth_loss' : True,
    'use_model_edge_loss' : True,
    'use_cd_loss' : True,

    'loss_edge_weight' : 1.0,
    'loss_laplacian_weight' : 1.0,
    'loss_normal_weight' : 0.01,
    'loss_silhouette_weight' : 1.0,
    'loss_depth_weight' : 1.0,
    'loss_model_edge_weight' : 1.0,
    'loss_cd_weight' : 1.0, 
}