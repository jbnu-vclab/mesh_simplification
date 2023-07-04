from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance, _FacePointDistance, _DEFAULT_MIN_TRIANGLE_AREA

point_face_distance = _PointFaceDistance.apply
face_point_distance = _FacePointDistance.apply

def point_mesh_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = face_to_point * weights_t
    face_dist = face_to_point.sum() / N

    return point_dist + face_dist