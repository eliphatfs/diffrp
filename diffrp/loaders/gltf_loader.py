import numpy
import torch
import logging
import trimesh
from PIL import Image
from typing import List, Union, BinaryIO
from trimesh.visual.material import PBRMaterial, SimpleMaterial
from trimesh.visual import ColorVisuals, TextureVisuals

from ..utils import colors
from ..utils.shader_ops import *
from ..scene import Scene, MeshObject
from ..rendering.camera import PerspectiveCamera
from ..materials.gltf_material import GLTFMaterial, GLTFSampler
from ._trimesh_gltf import load_glb as _load_glb, load_gltf as _load_gltf


def ensure_mode(img: Image.Image, mode: str):
    if img.mode == mode:
        return img
    return img.convert(mode)


def force_rgba(color: torch.Tensor):
    if (color > 1.5).any():
        color = color / 255.0
    if color.shape[-1] == 3:
        color = float4(color, 1.0)
    return color


def _simple_to_pbr(mat: SimpleMaterial):
    specular = mat.specular.astype(numpy.float32) / 255.0
    specular_intensity = specular[0] * 0.2125 + specular[1] * 0.7154 + specular[2] * 0.0721
    roughness = (2 / (mat.glossiness + 2)) ** (1.0 / 4.0)

    return PBRMaterial(
        roughnessFactor=roughness,
        baseColorTexture=mat.image,
        baseColorFactor=mat.diffuse,
        metallicFactor=specular_intensity,
    )


def to_gltf_material(verts: torch.Tensor, visual, material_cache: dict = {}, allow_convert=False):
    # GLTF 2.0 specifications 3.9.6 and 5.19
    default_mat = GLTFMaterial(
        gpu_f32([1, 1, 1, 1]),
        GLTFSampler(white_tex()),
        1.0, 1.0,
        GLTFSampler(white_tex()),
        None,
        None,
        None,
        GLTFSampler(white_tex().rgb),
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
        if allow_convert and isinstance(mat, SimpleMaterial):
            mat = _simple_to_pbr(mat)
        else:
            assert isinstance(mat, PBRMaterial), type(mat)
        if id(mat) in material_cache:
            return uv, color, material_cache[id(mat)]
        if mat.baseColorFactor is not None:
            default_mat.base_color_factor = force_rgba(gpu_f32(mat.baseColorFactor))
        if mat.baseColorTexture is not None:
            base_map = gpu_f32(numpy.asanyarray(ensure_mode(mat.baseColorTexture, "RGBA")))
            base_map /= 255.0
            base_map = float4(colors.srgb_to_linear(base_map.rgb), base_map.a)
            default_mat.base_color_texture.image = base_map
        if mat.metallicFactor is not None:
            default_mat.metallic_factor = float(mat.metallicFactor)
        if mat.roughnessFactor is not None:
            default_mat.roughness_factor = float(mat.roughnessFactor)
        if mat.metallicRoughnessTexture is not None:
            mr = gpu_f32(numpy.asanyarray(ensure_mode(mat.metallicRoughnessTexture, "RGB")))
            mr /= 255.0
            default_mat.metallic_roughness_texture.image = mr
        if mat.normalTexture is not None:
            nm = gpu_f32(numpy.asanyarray(ensure_mode(mat.normalTexture, "RGB")))
            nm /= 255.0
            default_mat.normal_texture = GLTFSampler(nm)
        if mat.occlusionTexture is not None:
            occ = gpu_f32(numpy.asanyarray(ensure_mode(mat.occlusionTexture, "RGB"))[..., :1])
            occ /= 255.0
            default_mat.occlusion_texture = GLTFSampler(occ)
        if mat.emissiveFactor is not None and (mat.emissiveFactor > 0).any():
            default_mat.emissive_factor = gpu_f32(mat.emissiveFactor)
        if mat.emissiveTexture is not None:
            emit = gpu_f32(numpy.asanyarray(ensure_mode(mat.emissiveTexture, "RGB")))
            emit /= 255.0
            emit = colors.srgb_to_linear(emit)
            default_mat.emissive_texture.image = emit
        if mat.alphaCutoff is not None:
            default_mat.alpha_cutoff = float(mat.alphaCutoff)
        if mat.alphaMode is not None:
            default_mat.alpha_mode = str(mat.alphaMode)
        material_cache[id(mat)] = default_mat
    else:
        assert False, ["Unknown visual", type(visual)]
    return uv, color, default_mat


def from_trimesh_scene(scene: Union[trimesh.Trimesh, trimesh.Scene], compute_tangents=False, allow_convert=True) -> Scene:
    """
    Convert a trimesh.Trimesh or trimesh.Scene to a DiffRP Scene.
    
    Supported metadata:
        ``name`` for each mesh object (may be ``None``);
        ``camera`` (type PerspectiveCamera) if exists.
    
    Args:
        scene (trimesh.Trimesh | trimesh.Scene): A trimesh loaded scene or mesh.
        compute_tangents (bool):
            If set, tangents will be computed according to the *MikkTSpace* algorithm.
            Execution of the algorithm requires ``gcc`` in the path.
            Defaults to ``False``.
            Note that computing tangents are not thread-safe.
        allow_convert (bool):
            DiffRP always try to interpret the materials as PBR materials.
            If set, allow a default conversion from simple materials (e.g., from obj and ply formats)
            to PBR materials.
        
    Returns:
        The loaded scene.
    """
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    drp_scene = Scene()
    if scene.has_camera:
        camera = PerspectiveCamera(
            float(scene.camera.fov[1]),
            round(float(scene.camera.resolution[1])),
            round(float(scene.camera.resolution[0])),
            float(scene.camera.z_near),
            float(scene.camera.z_far)
        )
        camera.t = scene.camera_transform
        drp_scene.metadata['camera'] = camera
    meshes: List[trimesh.Trimesh] = []
    transforms: List[torch.Tensor] = []
    names: List[str] = []
    discarded = 0
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        # get a copy of the geometry
        current = scene.geometry[geometry_name]
        if isinstance(current, trimesh.Trimesh):
            meshes.append(current)
            transforms.append(gpu_f32(transform))
            names.append(geometry_name)
    logging.info("Loaded scene with %d submeshes and %d discarded curve/pcd geometry", len(meshes), discarded)
    material_cache = {}
    for transform, mesh, name in zip(transforms, meshes, names):
        verts = gpu_f32(mesh.vertices)
        uv, color, mat = to_gltf_material(verts, mesh.visual, material_cache, allow_convert)
        # TODO: load vertex tangents if existing
        if 'vertex_normals' in mesh._cache and not compute_tangents:
            drp_scene.add_mesh_object(MeshObject(
                mat, verts,
                gpu_i32(mesh.faces),
                gpu_f32(mesh.vertex_normals),
                transform, color, uv,
                metadata=dict(name=name)
            ))
        else:
            flat_idx = mesh.faces.reshape(-1)
            # GLTF 2.0 specifications 3.7.2.1
            # if normal does not exist in data
            # client must calculate *flat* normals
            if 'vertex_normals' in mesh._cache:
                normals = mesh.vertex_normals[flat_idx]
            else:
                normals = numpy.stack([mesh.face_normals] * 3, axis=1).reshape(-1, 3)
            drp_scene.add_mesh_object(MeshObject(
                mat, verts[flat_idx],
                gpu_i32(numpy.arange(len(flat_idx)).reshape(-1, 3)),
                gpu_f32(normals),
                transform, color[flat_idx], uv[flat_idx],
                metadata=dict(name=name)
            ))
    if compute_tangents:
        from ..plugins import mikktspace
        for mesh_obj in drp_scene.objects:
            mesh_obj.tangents = mikktspace.execute(mesh_obj.verts, mesh_obj.normals, mesh_obj.uv)
    return drp_scene


def load_gltf_scene(path: Union[str, BinaryIO], compute_tangents=False) -> Scene:
    """
    Load a glb file as a DiffRP Scene.
    
    Supported metadata:
        ``name`` for each mesh object (may be ``None``);
        ``camera`` (type PerspectiveCamera) if exists.
    
    Args:
        path (str | BinaryIO): path to a ``.glb``/``.gltf`` file, or opened ``.glb`` file in binary mode.
        compute_tangents (bool):
            If set, tangents will be computed according to the *MikkTSpace* algorithm.
            Execution of the algorithm requires ``gcc`` in the path.
            Defaults to ``False``.
            Note that computing tangents are not thread-safe.
        
    Returns:
        The loaded scene.
    """
    loader = _load_glb
    if isinstance(path, str):
        if path.endswith(".gltf"):
            loader = _load_gltf
        path = open(path, "rb")
    kw = loader(path)
    path.close()
    scene: trimesh.Scene = trimesh.load(kw, force='scene', process=False)
    return from_trimesh_scene(scene, compute_tangents, False)
