import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh.shader import ShaderBase

# copied from pytorch3d shader, because of its bug

class HardDepthShader(ShaderBase):
    """
    Renders the Z distances of the closest face for each pixel. If no face is
    found it returns the zfar value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        for i in range(zfar.shape[0]):
            zbuf.masked_fill(mask[i, ...], zfar.unsqueeze(-1)[i].item())
        return zbuf

class SoftDepthShader(ShaderBase):
    """
    Renders the Z distances using an aggregate of the distances of each face
    based off of the point distance.  If no face is found it returns the zfar
    value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        if fragments.dists is None:
            raise ValueError("SoftDepthShader requires Fragments.dists to be present.")

        cameras = super()._get_cameras(**kwargs)

        N, H, W, K = fragments.pix_to_face.shape
        device = fragments.zbuf.device
        mask = fragments.pix_to_face >= 0

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 10.0))

        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / self.blend_params.sigma) * mask

        # append extra face for zfar
        b_dim = fragments.dists.shape[0]
        a = zfar.unsqueeze(-1).expand(b_dim,W).unsqueeze(-1).expand(b_dim,W,H).unsqueeze(-1)


        # zfars = torch.ones((N, H, W, 1), device=device) * zfar
        dists = torch.cat(
            (fragments.zbuf, a), dim=3
        )
        probs = torch.cat((prob_map, torch.ones((N, H, W, 1), device=device)), dim=3)

        # compute weighting based off of probabilities using cumsum
        probs = probs.cumsum(dim=3)
        probs = probs.clamp(max=1)
        probs = probs.diff(dim=3, prepend=torch.zeros((N, H, W, 1), device=device))

        image = (probs * dists).sum(dim=3).unsqueeze(3)
        normalized_image = (image - znear) / (zfar - znear)

        return normalized_image