import os
import torch
import ctypes
import subprocess


wd = os.path.dirname(os.path.abspath(__file__))
plugin_name = 'mikktspace_v1.bin'


def compile_plugin():
    subprocess.check_call([
        "gcc",
        "mikktinterface.c", "mikktspace.c",
        "-shared", "-fPIC", "-Ofast",
        "-o", plugin_name
    ], cwd=wd)


if not os.path.exists(os.path.join(wd, plugin_name)):
    compile_plugin()


mikkt = ctypes.CDLL(os.path.join(wd, plugin_name))
mikkt.compute_tri_tangents.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong)
mikkt.compute_tri_tangents.restype = None


@torch.no_grad()
def execute(verts: torch.Tensor, normals: torch.Tensor, uv: torch.Tensor):
    """
    Execute the *MikkTSpace* algorithm to
    compute tangent vectors and bitangent signs, which form the tangent space.
    The algorithm is serial algorithm on CPU and not differentiable,
    but tensors are automatically transferred to and back from CPU.

    It takes flat, unrolled vertex positions, normals and uvs.

    The plugin is not thread-safe.

    Args:
        verts (torch.Tensor): (F * 3, 3) vertex positions of faces unrolled.
        normals (torch.Tensor): (F * 3, 3) vertex normals of faces unrolled.
        uv (torch.Tensor): (F * 3, 2) vertex UV coordinates of faces unrolled.

    Returns:
        torch.Tensor: (F * 3, 4) tangent space. Has the same device and dtype as ``verts``.
    """
    vbuf = torch.cat([verts, normals, uv], dim=-1).to('cpu', torch.float32).contiguous()
    tangent = torch.empty([len(verts), 4], dtype=torch.float32, memory_format=torch.contiguous_format)
    mikkt.compute_tri_tangents(vbuf.data_ptr(), tangent.data_ptr(), len(vbuf))
    return tangent.to(verts)
