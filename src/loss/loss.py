import numpy as np

def loss_with_random_permutation(num_views,
                                 num_views_per_iteration,
                                 renderer,
                                 new_src_mesh,
                                 cameras,
                                 lights,
                                 target_imgs,
                                 loss_type:str,
                                 target_channel=3):
    if loss_type == 'mse':
        loss_func = mse_loss
    elif loss_type == 'iou':
        loss_func = iou_loss

    final_loss = 0
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images_predicted = renderer(new_src_mesh, cameras=cameras[j], lights=lights)
        predicted = images_predicted[..., target_channel]
        target = target_imgs[j, ..., target_channel]
            
        loss = loss_func(predicted, target)
        final_loss += loss / num_views_per_iteration

    return final_loss

def mse_loss(predicted, target):
    return ((predicted - target) ** 2).mean()

#  https://github.com/ShichenLiu/SoftRas
def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

# https://github.com/ShichenLiu/SoftRas
def iou_loss(predict, target):
    return 1 - iou(predict, target)

# https://github.com/ShichenLiu/SoftRas
def multiview_iou_loss(predicts, targets_a, targets_b):
    loss = (iou_loss(predicts[0][:, 3], targets_a[:, 3]) +
            iou_loss(predicts[1][:, 3], targets_a[:, 3]) +
            iou_loss(predicts[2][:, 3], targets_b[:, 3]) +
            iou_loss(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss
