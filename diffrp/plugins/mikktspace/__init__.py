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
    vbuf = torch.cat([verts, normals, uv], dim=-1).to('cpu', torch.float32).contiguous()
    tangent = torch.empty([len(verts), 4], dtype=torch.float32, memory_format=torch.contiguous_format)
    mikkt.compute_tri_tangents(vbuf.data_ptr(), tangent.data_ptr(), len(vbuf))
    return tangent.to(verts)
