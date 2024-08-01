import torch
from .shader_ops import normalized, cross


def make_face_soup(verts, tris, face_normals):
    return (
        verts[tris].reshape(-1, 3),
        torch.arange(len(tris.reshape(-1)), device=tris.device, dtype=torch.int32).reshape(tris.shape),
        torch.stack([face_normals] * 3, axis=1).reshape(-1, 3)
    )


def compute_face_normals(verts: torch.Tensor, faces: torch.Tensor, normalize=False):
    face_normals = cross(verts[faces[:, 1]] - verts[faces[:, 0]], 
                         verts[faces[:, 2]] - verts[faces[:, 0]])
    return normalized(face_normals) if normalize else face_normals


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor):
    """
    Compute per-vertex normals from vertices and faces.
    
    Args:
        verts (torch.Tensor): Tensor of shape (num_verts, 3) containing vertex coordinates.
        faces (torch.Tensor): Tensor of shape (num_faces, 3) containing triangle indices.
        
    Returns:
        torch.Tensor: Tensor of shape (num_verts, 3) containing per-vertex normals.
    """
    face_normals = compute_face_normals(verts, faces)
    
    vertex_normals = torch.zeros_like(verts)
    vertex_normals = vertex_normals.index_add(0, faces[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 1], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 2], face_normals)
    
    vertex_normals = normalized(vertex_normals)
    
    return vertex_normals
