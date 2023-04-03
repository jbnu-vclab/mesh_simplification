import torch
import torch.nn.functional as F
import pytorch3d
from pytorch3d import renderer
from pytorch3d.renderer.mesh.shader import ShaderBase, phong_shading
import torch
from pytorch3d.renderer import SoftPhongShader
from pytorch3d.renderer.mesh.shader import ShaderBase

class ModelEdgeShader(ShaderBase):
    def __init__(self, device="cpu", edge_threshold=0.0002):
        super().__init__()
        self.device = device
        self.edge_threshold = edge_threshold


    def forward(self, fragments, meshes, **kwargs):
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = 4 # (b x W x H x c)

        # Depth 값으로부터 2D Laplace 필터로 Edge 계산
        zbuf = zbuf.permute(0,3,1,2) # (bxcxWxH)

        laplace_2d_filter = torch.tensor([[1., 1., 1.],
                                          [1.,-8., 1.],
                                          [1., 1., 1.]], device=zbuf.device) # (3x3)
        laplace_2d_filter = laplace_2d_filter.unsqueeze(0).unsqueeze(0) # (1x1x3x3) # TODO: Batch 고려 필요

        gaussian_filter = torch.tensor([[1.,  4.,  7.,  4., 1.],
                                        [4., 16., 26., 16., 4.],
                                        [7., 26., 41., 26., 7.],
                                        [4., 16., 26., 16., 4.],
                                        [1.,  4.,  7.,  4., 1.]], device=zbuf.device) / 273
        gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)

        edge_zbuf = F.conv2d(zbuf, laplace_2d_filter, padding='same')
        edge_img = (edge_zbuf >= self.edge_threshold).float()

        blurred_img = F.conv2d(edge_img, gaussian_filter, padding='same')
        blurred_img = F.conv2d(edge_img, gaussian_filter, padding='same')
        blurred_img = blurred_img + edge_img
        blurred_img = F.conv2d(blurred_img, gaussian_filter, padding='same')
        blurred_img = F.conv2d(blurred_img, gaussian_filter, padding='same')
        blurred_img = blurred_img + edge_img

        image = blurred_img.permute(0, 2, 3, 1) # (bxWxHxc)

        return image

class SimpleEdgeShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = 4 # (b x W x H x c)

        # Depth 값으로부터 2D Laplace 필터로 Edge 계산
        zbuf = zbuf.permute(0,3,1,2) # (bxcxWxH)

        laplace_2d_filter = torch.tensor([[1., 1., 1.],
                                          [1.,-8., 1.],
                                          [1., 1., 1.]], device=zbuf.device) # (3x3)
        laplace_2d_filter = laplace_2d_filter.unsqueeze(0).unsqueeze(0) # (1x1x3x3) # TODO: Batch 고려 필요

        edge_zbuf = F.conv2d(zbuf, laplace_2d_filter, padding='same')
        edge_img = (edge_zbuf >= 0.02).float()

        image = edge_img.permute(0, 2, 3, 1) # (bxWxHxc)

        return image