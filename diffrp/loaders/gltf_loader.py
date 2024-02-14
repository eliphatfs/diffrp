import numpy
import torch
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import PBRMaterial
from dataclasses import dataclass
from typing import List
import logging

from .. import colors
from ..shader_ops import *
from ..materials.gltf_material import GLTFMaterial, GLTFSampler


@dataclass
class GLTFMeshPrimitive:
    material: GLTFMaterial
    verts: torch.Tensor
    tris: torch.IntTensor
    normals: torch.Tensor
    uv: torch.Tensor
    color: torch.Tensor


def force_rgba(color: torch.Tensor):
    if (color > 2.0).any():
        color = color / 255.0
    if color.shape[-1] == 3:
        color = float4(color, 1.0)
    return color


def to_gltf_material(verts: torch.Tensor, visual):
    # GLTF 2.0 specifications 3.9.6 and 5.19
    default_mat = GLTFMaterial(
        gpu_f32([1, 1, 1, 1]),
        GLTFSampler(white_tex),
        1.0, 1.0,
        GLTFSampler(white_tex),
        GLTFSampler(empty_normal_tex),
        GLTFSampler(white_tex),
        gpu_f32([0, 0, 0]),
        GLTFSampler(black_tex.rgb),
        0.5, 'OPAQUE'
    )
    color = ones_like_vec(verts, 4)
    uv = zeros_like_vec(verts, 2)
    if isinstance(visual, ColorVisuals):
        color = force_rgba(gpu_f32(visual.vertex_colors))
    elif isinstance(visual, TextureVisuals):
        if 'color' in visual.vertex_attributes:
            color = force_rgba(gpu_f32(visual.vertex_attributes['color']))
        if visual.uv is not None:
            uv = gpu_f32(visual.uv)
        mat = visual.material
        assert isinstance(mat, PBRMaterial), type(mat)
        if mat.baseColorFactor is not None:
            default_mat.base_color_factor = force_rgba(gpu_f32(mat.baseColorFactor))
        if mat.baseColorTexture is not None:
            srgb = gpu_f32(numpy.array(mat.baseColorTexture.convert("RGBA"))) / 255.0
            default_mat.base_color_texture.image = float4(colors.srgb_to_linear(srgb.rgb), srgb.a)
        if mat.metallicFactor is not None:
            default_mat.metallic_factor = float(mat.metallicFactor)
        if mat.roughnessFactor is not None:
            default_mat.roughness_factor = float(mat.roughnessFactor)
        if mat.metallicRoughnessTexture is not None:
            mr = gpu_f32(numpy.array(mat.metallicRoughnessTexture.convert("RGB"))) / 255.0
            default_mat.metallic_roughness_texture.image = mr
        if mat.normalTexture is not None:
            nm = gpu_f32(numpy.array(mat.normalTexture.convert("RGB"))) / 255.0
            default_mat.normal_texture.image = nm
        if mat.occlusionTexture is not None:
            occ = gpu_f32(numpy.array(mat.occlusionTexture.convert("RGB"))) / 255.0
            default_mat.occlusion_texture.image = occ
        if mat.emissiveFactor is not None:
            default_mat.emissive_factor = gpu_f32(mat.emissiveFactor)
        if mat.emissiveTexture is not None:
            emit = gpu_f32(numpy.array(mat.emissiveTexture.convert("RGB"))) / 255.0
            default_mat.emissive_texture.image = colors.srgb_to_linear(emit)
        if mat.alphaCutoff is not None:
            default_mat.alpha_cutoff = float(mat.alphaCutoff)
        if mat.alphaMode is not None:
            default_mat.alpha_mode = str(mat.alphaMode)
    else:
        assert False, ["Unknown visual", type(visual)]
    return uv, color, default_mat


class GLTFLoader:
    prims: List[GLTFMeshPrimitive]

    def __init__(self, path) -> None:
        scene: trimesh.Scene = trimesh.load(path, force='scene', process=False)
        prims = scene.dump(concatenate=False)
        meshes: List[trimesh.Trimesh] = [f for f in prims if isinstance(f, trimesh.Trimesh)]
        discarded = [f for f in prims if not isinstance(f, trimesh.Trimesh)]
        logging.info("Loaded scene %s with %d submeshes and %d discarded curve/pcd geometry", path, len(meshes), len(discarded))
        self.prims = []
        for mesh in meshes:
            verts = gpu_f32(mesh.vertices)
            uv, color, mat = to_gltf_material(verts, mesh.visual)
            if 'vertex_normals' in mesh._cache:
                self.prims.append(GLTFMeshPrimitive(
                    mat, verts,
                    gpu_i32(mesh.faces),
                    gpu_f32(mesh.vertex_normals),
                    uv, color
                ))
            else:
                # GLTF 2.0 specifications 3.7.2.1
                # if normal does not exist in data
                # client must calculate *flat* normals
                flat_idx = mesh.faces.reshape(-1)
                self.prims.append(GLTFMeshPrimitive(
                    mat, verts[flat_idx],
                    gpu_i32(numpy.arange(len(flat_idx)).reshape(-1, 3)),
                    gpu_f32(numpy.stack([mesh.face_normals] * 3, axis=1).reshape(-1, 3)),
                    uv[flat_idx], color[flat_idx]
                ))
