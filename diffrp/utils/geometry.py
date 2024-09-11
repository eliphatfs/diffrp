import torch
from .shader_ops import normalized, cross


def make_face_soup(verts, tris, face_normals):
    """
    Make a face soup geometry from vertices, triangles and face normals.

    Args:
        verts (torch.Tensor): Tensor of shape (num_verts, 3) containing vertex coordinates.
        faces (torch.Tensor): Tensor of shape (num_faces, 3) containing triangle indices.
        face_normals (torch.Tensor): Tensor of shape (num_faces, 3) containing face normals.
        
    Returns:
        tuple of (verts, tris, vertex_normals):
            Constructed geometry.
            There are 3 * num_faces vertices and num_faces triangles in the result.
    """
    return (
        verts[tris].reshape(-1, 3),
        torch.arange(len(tris.reshape(-1)), device=tris.device, dtype=torch.int32).reshape(tris.shape),
        torch.stack([face_normals] * 3, axis=1).reshape(-1, 3)
    )


def compute_face_normals(verts: torch.Tensor, faces: torch.Tensor, normalize=False):
    """
    Compute per-face normals from vertices and faces.
    Faces are assumed to be CCW winding.

    Args:
        verts (torch.Tensor): Tensor of shape (num_verts, 3) containing vertex coordinates.
        faces (torch.Tensor): Tensor of shape (num_faces, 3) containing triangle indices.
        normalize (bool): Whether the results should be normalized.
        
    Returns:
        torch.Tensor: Tensor of shape (num_faces, 3) containing per-face normals.
    """
    face_normals = cross(verts[faces[:, 1]] - verts[faces[:, 0]], 
                         verts[faces[:, 2]] - verts[faces[:, 0]])
    return normalized(face_normals) if normalize else face_normals


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor):
    """
    Compute per-vertex normals from vertices and faces.
    Faces are assumed to be CCW winding.
    
    Args:
        verts (torch.Tensor): Tensor of shape (num_verts, 3) containing vertex coordinates.
        faces (torch.Tensor): Tensor of shape (num_faces, 3) containing triangle indices.
        
    Returns:
        torch.Tensor: Tensor of shape (num_verts, 3) containing normalized per-vertex normals.
    """
    face_normals = compute_face_normals(verts, faces)
    
    vertex_normals = torch.zeros_like(verts)
    vertex_normals = vertex_normals.index_add(0, faces[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 1], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 2], face_normals)
    
    vertex_normals = normalized(vertex_normals)
    
    return vertex_normals


def sign2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Modified from ``calibur.sign2d``.
    """
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)


def point_in_tri2d(pt: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
    """
    Decide whether points lie in 2D triangles. Tensors broadcast.
    Modified from ``calibur.point_in_tri2d``.

    :param pt: ``(..., 2)`` points to decide.
    :param v1: ``(..., 2)`` first vertex in triangles.
    :param v2: ``(..., 2)`` second vertex in triangles.
    :param v3: ``(..., 2)`` third vertex in triangles.
    :returns: ``(..., 1)`` boolean result of points lie in triangles.
    """
    d1 = sign2d(pt, v1, v2)
    d2 = sign2d(pt, v2, v3)
    d3 = sign2d(pt, v3, v1)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


@torch.jit.script
def _barycentric_impl(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: torch.Tensor):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    v = torch.clip(torch.nan_to_num(v), 0, 1)
    w = torch.clip(torch.nan_to_num(w), 0, 1)
    u = 1 - v - w
    return torch.stack([u, v, w], dim=-1)


def _cancel_nan(x):
    return torch.nan_to_num(x)


def _cancel_nans(*tensors: torch.Tensor):
    for tnsr in tensors:
        if tnsr.requires_grad:
            tnsr.register_hook(_cancel_nan)


def barycentric(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: torch.Tensor):
    """
    Computes barycentric coordinates from a point ``p`` and triangle vertices ``a``, ``b`` and ``c``.

    Args:
        Takes broadcastable inputs of (..., 3).
    
    Returns:
        torch.Tensor: Stacked ``(u, v, w)`` tensor of shape (..., 3) s.t. ``p = ua + vb + wc``.
    """
    _cancel_nans(a, b, c, p)
    return _barycentric_impl(a, b, c, p)
