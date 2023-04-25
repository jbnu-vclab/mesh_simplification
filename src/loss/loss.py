import numpy as np

def loss_with_random_permutation(num_views, num_views_per_iteration, renderer, new_src_mesh, cameras, lights, target_imgs, loss_func, target_channel=3):
    final_loss = 0
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images_predicted = renderer(new_src_mesh, cameras=cameras[j], lights=lights)
        predicted = images_predicted[..., target_channel]
        target = target_imgs[j, ..., target_channel]
            
        loss = loss_func(predicted, target)
        final_loss += loss / num_views_per_iteration

    return final_loss

def mse_loss(predicted_img, target_img):
    return ((predicted_img - target_img) ** 2).mean()
    
