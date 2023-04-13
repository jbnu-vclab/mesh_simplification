import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh.shader import ShaderBase

class DepthMapShader(ShaderBase):

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf