"""
gltf.py
------------

Provides GLTF 2.0 exports of trimesh.Trimesh objects
as GL_TRIANGLES, and trimesh.Path2D/Path3D as GL_LINES

Copied and modified from trimesh to support a more complete set in GLTF2
"""

import base64
import json
from collections import OrderedDict, defaultdict, deque

import numpy as np

from trimesh import transformations, util, visual
from trimesh.constants import log
from trimesh.resolvers import Resolver
from trimesh.scene.cameras import Camera
from trimesh.typed import Mapping, Optional, Stream, Union
from trimesh.util import triangle_strips_to_faces, unique_name
from trimesh.visual.gloss import specular_to_pbr

# magic numbers which have meaning in GLTF
# most are uint32's of UTF-8 text
_magic = {"gltf": 1179937895, "json": 1313821514, "bin": 5130562}

# GLTF data type codes: little endian numpy dtypes
_dtypes = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
# a string we can use to look up numpy dtype : GLTF dtype
_dtypes_lookup = {v[1:]: k for k, v in _dtypes.items()}

# GLTF data formats: numpy shapes
_shapes = {
    "SCALAR": 1,
    "VEC2": (2),
    "VEC3": (3),
    "VEC4": (4),
    "MAT2": (2, 2),
    "MAT3": (3, 3),
    "MAT4": (4, 4),
}

# a default PBR metallic material
_default_material = {
    "pbrMetallicRoughness": {
        "baseColorFactor": [1, 1, 1, 1],
        "metallicFactor": 0,
        "roughnessFactor": 0,
    }
}

# we can accept dict resolvers
ResolverLike = Union[Resolver, Mapping]

# GL geometry modes
_GL_LINES = 1
_GL_POINTS = 0
_GL_TRIANGLES = 4
_GL_STRIP = 5

_EYE = np.eye(4)
_EYE.flags.writeable = False

# specify dtypes with forced little endian
float32 = np.dtype("<f4")
uint32 = np.dtype("<u4")
uint8 = np.dtype("<u1")


def load_gltf(
    file_obj: Optional[Stream] = None,
    resolver: Optional[ResolverLike] = None,
    ignore_broken: bool = False,
    merge_primitives: bool = False,
    skip_materials: bool = False,
    **mesh_kwargs,
):
    """
    Load a GLTF file, which consists of a directory structure
    with multiple files.

    Parameters
    -------------
    file_obj : None or file-like
      Object containing header JSON, or None
    resolver : trimesh.visual.Resolver
      Object which can be used to load other files by name
    ignore_broken : bool
      If there is a mesh we can't load and this
      is True don't raise an exception but return
      a partial result
    merge_primitives : bool
      If True, each GLTF 'mesh' will correspond
      to a single Trimesh object
    skip_materials : bool
      If true, will not load materials (if present).
    **mesh_kwargs : dict
      Passed to mesh constructor

    Returns
    --------------
    kwargs : dict
      Arguments to create scene
    """
    try:
        # see if we've been passed the GLTF header file
        tree = json.loads(util.decode_text(file_obj.read()))
    except BaseException:
        # otherwise header should be in 'model.gltf'
        data = resolver["model.gltf"]
        # old versions of python/json need strings
        tree = json.loads(util.decode_text(data))

    # gltf 1.0 is a totally different format
    # that wasn't widely deployed before they fixed it
    version = tree.get("asset", {}).get("version", "2.0")
    if isinstance(version, str):
        # parse semver like '1.0.1' into just a major integer
        major = int(version.split(".", 1)[0])
    else:
        major = int(float(version))

    if major < 2:
        raise NotImplementedError(f"only GLTF 2 is supported not `{version}`")

    # use the URI and resolver to get data from file names
    buffers = [
        _uri_to_bytes(uri=b["uri"], resolver=resolver) for b in tree.get("buffers", [])
    ]

    # turn the layout header and data into kwargs
    # that can be used to instantiate a trimesh.Scene object
    kwargs = _read_buffers(
        header=tree,
        buffers=buffers,
        ignore_broken=ignore_broken,
        merge_primitives=merge_primitives,
        mesh_kwargs=mesh_kwargs,
        skip_materials=skip_materials,
        resolver=resolver,
    )
    return kwargs


def load_glb(
    file_obj: Stream,
    resolver: Optional[ResolverLike] = None,
    ignore_broken: bool = False,
    merge_primitives: bool = False,
    skip_materials: bool = False,
    **mesh_kwargs,
):
    """
    Load a GLTF file in the binary GLB format into a trimesh.Scene.

    Implemented from specification:
    https://github.com/KhronosGroup/glTF/tree/master/specification/2.0

    Parameters
    ------------
    file_obj : file- like object
      Containing GLB data
    resolver : trimesh.visual.Resolver
      Object which can be used to load other files by name
    ignore_broken : bool
      If there is a mesh we can't load and this
      is True don't raise an exception but return
      a partial result
    merge_primitives : bool
      If True, each GLTF 'mesh' will correspond to a
      single Trimesh object.
    skip_materials : bool
      If true, will not load materials (if present).

    Returns
    ------------
    kwargs : dict
      Kwargs to instantiate a trimesh.Scene
    """
    # read the first 20 bytes which contain section lengths
    head_data = file_obj.read(20)
    head = np.frombuffer(head_data, dtype="<u4")

    # check to make sure first index is gltf magic header
    if head[0] != _magic["gltf"]:
        raise ValueError("incorrect header on GLB file")

    # and second value is version: should be 2 for GLTF 2.0
    if head[1] != 2:
        raise NotImplementedError(f"only GLTF 2 is supported not `{head[1]}`")

    # overall file length
    # first chunk length
    # first chunk type
    length, chunk_length, chunk_type = head[2:]

    # first chunk should be JSON header
    if chunk_type != _magic["json"]:
        raise ValueError("no initial JSON header!")

    # uint32 causes an error in read, so we convert to native int
    # for the length passed to read, for the JSON header
    json_data = file_obj.read(int(chunk_length))
    # convert to text
    if hasattr(json_data, "decode"):
        json_data = util.decode_text(json_data)
    # load the json header to native dict
    header = json.loads(json_data)

    # read the binary data referred to by GLTF as 'buffers'
    buffers = []
    start = file_obj.tell()

    # header can contain base64 encoded data in the URI field
    info = header.get("buffers", []).copy()

    while (file_obj.tell() - start) < length:
        # if we have buffer infos with URI check it here
        try:
            # if they have interleaved URI data with GLB data handle it here
            uri = info.pop(0)["uri"]
            buffers.append(_uri_to_bytes(uri=uri, resolver=resolver))
            continue
        except (IndexError, KeyError):
            # if there was no buffer info or URI we still need to read
            pass

        # the last read put us past the JSON chunk
        # we now read the chunk header, which is 8 bytes
        chunk_head = file_obj.read(8)
        if len(chunk_head) != 8:
            # double check to make sure we didn't
            # read the whole file
            break
        chunk_length, chunk_type = np.frombuffer(chunk_head, dtype="<u4")
        # make sure we have the right data type
        if chunk_type != _magic["bin"]:
            raise ValueError("not binary GLTF!")
        # read the chunk
        chunk_data = file_obj.read(int(chunk_length))
        if len(chunk_data) != chunk_length:
            raise ValueError("chunk was not expected length!")
        buffers.append(chunk_data)

    # turn the layout header and data into kwargs
    # that can be used to instantiate a trimesh.Scene object
    kwargs = _read_buffers(
        header=header,
        buffers=buffers,
        ignore_broken=ignore_broken,
        merge_primitives=merge_primitives,
        skip_materials=skip_materials,
        mesh_kwargs=mesh_kwargs,
        resolver=resolver,
    )

    return kwargs


def _uri_to_bytes(uri, resolver):
    """
    Take a URI string and load it as a
    a filename or as base64.

    Parameters
    --------------
    uri : string
      Usually a filename or something like:
      "data:object/stuff,base64,AABA112A..."
    resolver : trimesh.visual.Resolver
      A resolver to load referenced assets

    Returns
    ---------------
    data : bytes
      Loaded data from URI
    """
    # see if the URI has base64 data
    index = uri.find("base64,")
    if index < 0:
        # string didn't contain the base64 header
        # so return the result from the resolver
        return resolver[uri]
    # we have a base64 header so strip off
    # leading index and then decode into bytes
    return base64.b64decode(uri[index + 7 :])





def _parse_textures(header, views, resolver=None):
    try:
        import PIL.Image
    except ImportError:
        log.debug("unable to load textures without pillow!")
        return None

    # load any images
    images = None
    if "images" in header:
        # images are referenced by index
        images = [None] * len(header["images"])
        # loop through images
        for i, img in enumerate(header["images"]):
            # get the bytes representing an image
            if "bufferView" in img:
                blob = views[img["bufferView"]]
            elif "uri" in img:
                # will get bytes from filesystem or base64 URI
                blob = _uri_to_bytes(uri=img["uri"], resolver=resolver)
            else:
                log.debug(f"unable to load image from: {img.keys()}")
                continue
            # i.e. 'image/jpeg'
            # mime = img['mimeType']
            try:
                # load the buffer into a PIL image
                images[i] = PIL.Image.open(util.wrap_as_stream(blob))
            except BaseException:
                log.error("failed to load image!", exc_info=True)
    return images


def _parse_materials(header, views, resolver=None):
    """
    Convert materials and images stored in a GLTF header
    and buffer views to PBRMaterial objects.

    Parameters
    ------------
    header : dict
      Contains layout of file
    views : (n,) bytes
      Raw data

    Returns
    ------------
    materials : list
      List of trimesh.visual.texture.Material objects
    """

    def parse_values_and_textures(input_dict):
        result = {}
        for k, v in input_dict.items():
            if isinstance(v, (list, tuple)):
                # colors are always float 0.0 - 1.0 in GLTF
                result[k] = np.array(v, dtype=np.float64)
            elif not isinstance(v, dict):
                result[k] = v
            elif "index" in v:
                # get the index of image for texture

                try:
                    texture = header["textures"][v["index"]]

                    # check to see if this is using a webp extension texture
                    # should this be case sensitive?
                    webp = (
                        texture.get("extensions", {})
                        .get("EXT_texture_webp", {})
                        .get("source")
                    )
                    if webp is not None:
                        idx = webp
                    else:
                        # fallback (or primary, if extensions are not present)
                        idx = texture["source"]

                    # store the actual image as the value
                    result[k] = images[idx]
                except BaseException:
                    log.debug("unable to store texture", exc_info=True)
        return result

    images = _parse_textures(header, views, resolver)

    # store materials which reference images
    materials = []
    if "materials" in header:
        for mat in header["materials"]:
            # flatten key structure so we can loop it
            loopable = mat.copy()
            # this key stores another dict of crap
            if "pbrMetallicRoughness" in loopable:
                # add keys of keys to top level dict
                loopable.update(loopable.pop("pbrMetallicRoughness"))

            ext = mat.get("extensions", {}).get(
                "KHR_materials_pbrSpecularGlossiness", None
            )
            if isinstance(ext, dict):
                ext_params = parse_values_and_textures(ext)
                loopable.update(specular_to_pbr(**ext_params))

            # save flattened keys we can use for kwargs
            pbr = parse_values_and_textures(loopable)
            # create a PBR material object for the GLTF material
            materials.append(visual.material.PBRMaterial(**pbr))

    return materials


def _read_buffers(
    header,
    buffers,
    mesh_kwargs,
    resolver: Optional[ResolverLike],
    ignore_broken: bool = False,
    merge_primitives: bool = False,
    skip_materials: bool = False,
):
    """
    Given binary data and a layout return the
    kwargs to create a scene object.

    Parameters
    -----------
    header : dict
      With GLTF keys
    buffers : list of bytes
      Stored data
    mesh_kwargs : dict
      To be passed to the mesh constructor.
    ignore_broken : bool
      If there is a mesh we can't load and this
      is True don't raise an exception but return
      a partial result
    merge_primitives : bool
      If true, combine primitives into a single mesh.
    skip_materials : bool
      If true, will not load materials (if present).
    resolver : trimesh.resolvers.Resolver
      Resolver to load referenced assets

    Returns
    -----------
    kwargs : dict
      Can be passed to load_kwargs for a trimesh.Scene
    """

    if "bufferViews" in header:
        # split buffer data into buffer views
        views = [None] * len(header["bufferViews"])
        for i, view in enumerate(header["bufferViews"]):
            if "byteOffset" in view:
                start = view["byteOffset"]
            else:
                start = 0
            end = start + view["byteLength"]
            views[i] = buffers[view["buffer"]][start:end]
            assert len(views[i]) == view["byteLength"]
        # load data from buffers into numpy arrays
        # using the layout described by accessors
        access = [None] * len(header["accessors"])
        for index, a in enumerate(header["accessors"]):
            # number of items
            count = a["count"]
            # what is the datatype
            dtype = np.dtype(_dtypes[a["componentType"]])
            # basically how many columns
            # for types like (4, 4)
            per_item = _shapes[a["type"]]
            # use reported count to generate shape
            shape = np.append(count, per_item)
            # number of items when flattened
            # i.e. a (4, 4) MAT4 has 16
            per_count = np.abs(np.prod(per_item))
            if "bufferView" in a:
                # data was stored in a buffer view so get raw bytes

                # load the bytes data into correct dtype and shape
                buffer_view = header["bufferViews"][a["bufferView"]]

                # is the accessor offset in a buffer
                # will include the start, length, and offset
                # but not the bytestride as that is easier to do
                # in numpy rather than in python looping
                data = views[a["bufferView"]]

                # both bufferView *and* accessors are allowed
                # to have a byteOffset
                start = a.get("byteOffset", 0)

                if "byteStride" in buffer_view:
                    # how many bytes for each chunk
                    stride = buffer_view["byteStride"]
                    # we want to get the bytes for every row
                    per_row = per_count * dtype.itemsize
                    # the total block we're looking at
                    length = (count - 1) * stride + per_row
                    # we have to offset the (already offset) buffer
                    # and then pull chunks per-stride
                    # do as a list comprehension as the numpy
                    # buffer wangling was

                    # raw = b"".join(
                    #     data[i : i + per_row]
                    #     for i in range(start, start + length, stride)
                    # )
                    # the reshape should fail if we screwed up
                    # access[index] = np.frombuffer(raw, dtype=dtype).reshape(shape)
                    assert stride > 0, "byteStride should be positive"
                    assert 0 <= start <= start + length <= len(data)
                    access[index] = np.array(
                        np.lib.stride_tricks.as_strided(
                            np.frombuffer(data, dtype=np.uint8, offset=start, count=length),
                            [count, per_row], [stride, 1]
                        ).view(dtype).reshape(shape)
                    )
                else:
                    # length is the number of bytes per item times total
                    length = dtype.itemsize * count * per_count
                    access[index] = np.frombuffer(
                        data[start : start + length], dtype=dtype
                    ).reshape(shape)
            else:
                # a "sparse" accessor should be initialized as zeros
                access[index] = np.zeros(count * per_count, dtype=dtype).reshape(shape)

        # possibly load images and textures into material objects
        if skip_materials:
            materials = []
        else:
            materials = _parse_materials(header, views=views, resolver=resolver)

    mesh_prim = defaultdict(list)
    # load data from accessors into Trimesh objects
    meshes = OrderedDict()

    # keep track of how many times each name has been attempted to
    # be inserted to avoid a potentially slow search through our
    # dict of names
    name_counts = {}
    for index, m in enumerate(header.get("meshes", [])):
        try:
            # GLTF spec indicates implicit units are meters
            metadata = {"units": "meters"}
            # try to load all mesh metadata
            if isinstance(m.get("extras"), dict):
                metadata.update(m["extras"])
            # put any mesh extensions in a field of the metadata
            if "extensions" in m:
                metadata["gltf_extensions"] = m["extensions"]

            for p in m["primitives"]:
                # if we don't have a triangular mesh continue
                # if not specified assume it is a mesh
                kwargs = {"metadata": {}, "process": False}
                kwargs.update(mesh_kwargs)
                kwargs["metadata"].update(metadata)
                # i.e. GL_LINES, GL_TRIANGLES, etc
                # specification says the default mode is GL_TRIANGLES
                mode = p.get("mode", _GL_TRIANGLES)
                # colors, normals, etc
                attr = p["attributes"]
                # create a unique mesh name per- primitive
                name = m.get("name", "GLTF")
                # make name unique across multiple meshes
                name = unique_name(name, meshes, counts=name_counts)

                if mode == _GL_LINES:
                    # load GL_LINES into a Path object
                    from trimesh.path.entities import Line

                    kwargs["vertices"] = access[attr["POSITION"]]
                    kwargs["entities"] = [Line(points=np.arange(len(kwargs["vertices"])))]
                elif mode == _GL_POINTS:
                    kwargs["vertices"] = access[attr["POSITION"]]
                elif mode in (_GL_TRIANGLES, _GL_STRIP):
                    # get vertices from accessors
                    kwargs["vertices"] = access[attr["POSITION"]]
                    # get faces from accessors
                    if "indices" in p:
                        if mode == _GL_STRIP:
                            # this is triangle strips
                            flat = access[p["indices"]].reshape(-1)
                            kwargs["faces"] = triangle_strips_to_faces([flat])
                        else:
                            kwargs["faces"] = access[p["indices"]].reshape((-1, 3))
                    else:
                        # indices are apparently optional and we are supposed to
                        # do the same thing as webGL drawArrays?
                        if mode == _GL_STRIP:
                            kwargs["faces"] = triangle_strips_to_faces(
                                np.array([np.arange(len(kwargs["vertices"]))])
                            )
                        else:
                            # GL_TRIANGLES
                            kwargs["faces"] = np.arange(
                                len(kwargs["vertices"]), dtype=np.int64
                            ).reshape((-1, 3))

                    if "NORMAL" in attr:
                        # vertex normals are specified
                        kwargs["vertex_normals"] = access[attr["NORMAL"]]
                        # do we have UV coordinates
                    visuals = None
                    if "material" in p and not skip_materials:
                        if materials is None:
                            log.debug("no materials! `pip install pillow`")
                        else:
                            uv = None
                            if "TEXCOORD_0" in attr:
                                # flip UV's top- bottom to move origin to lower-left:
                                # https://github.com/KhronosGroup/glTF/issues/1021
                                uv = access[attr["TEXCOORD_0"]].copy()
                                uv[:, 1] = 1.0 - uv[:, 1]
                                # create a texture visual
                            visuals = visual.texture.TextureVisuals(
                                uv=uv, material=materials[p["material"]]
                            )

                    if "COLOR_0" in attr:
                        try:
                            # try to load vertex colors from the accessors
                            colors = access[attr["COLOR_0"]]
                            if len(colors) == len(kwargs["vertices"]):
                                if visuals is None:
                                    # just pass to mesh as vertex color
                                    kwargs["vertex_colors"] = colors.copy()
                                else:
                                    # we ALSO have texture so save as vertex
                                    # attribute
                                    visuals.vertex_attributes["color"] = colors.copy()
                        except BaseException:
                            # survive failed colors
                            log.debug("failed to load colors", exc_info=True)
                    if visuals is not None:
                        kwargs["visual"] = visuals

                    # By default the created mesh is not from primitive,
                    # in case it is the value will be updated
                    # each primitive gets it's own Trimesh object
                    if len(m["primitives"]) > 1:
                        kwargs["metadata"]["from_gltf_primitive"] = True
                    else:
                        kwargs["metadata"]["from_gltf_primitive"] = False

                    # custom attributes starting with a `_`
                    custom = {
                        a: access[attr[a]] for a in attr.keys() if a.startswith("_")
                    }
                    if len(custom) > 0:
                        kwargs["vertex_attributes"] = custom
                else:
                    log.debug("skipping primitive with mode %s!", mode)
                    continue
                # this should absolutely not be stomping on itself
                assert name not in meshes
                meshes[name] = kwargs
                mesh_prim[index].append(name)
        except BaseException as E:
            if ignore_broken:
                log.debug("failed to load mesh", exc_info=True)
            else:
                raise E

    # sometimes GLTF "meshes" come with multiple "primitives"
    # by default we return one Trimesh object per "primitive"
    # but if merge_primitives is True we combine the primitives
    # for the "mesh" into a single Trimesh object
    if merge_primitives:
        # if we are only returning one Trimesh object
        # replace `mesh_prim` with updated values
        mesh_prim_replace = {}
        # these are the names of meshes we need to remove
        mesh_pop = set()
        for mesh_index, names in mesh_prim.items():
            if len(names) <= 1:
                mesh_prim_replace[mesh_index] = names
                continue

            # just take the shortest name option available
            name = min(names)
            # remove the other meshes after we're done looping
            # since we're reusing the shortest one don't pop
            # that as we'll be overwriting it with the combined
            mesh_pop.update(set(names).difference([name]))

            # get all meshes for this group
            current = [meshes[n] for n in names]
            v_seq = [p["vertices"] for p in current]
            f_seq = [p["faces"] for p in current]
            v, f = util.append_faces(v_seq, f_seq)
            materials = [p["visual"].material for p in current]
            face_materials = []
            for i, p in enumerate(current):
                face_materials += [i] * len(p["faces"])
            visuals = visual.texture.TextureVisuals(
                material=visual.material.MultiMaterial(materials=materials),
                face_materials=face_materials,
            )
            if "metadata" in meshes[names[0]]:
                metadata = meshes[names[0]]["metadata"]
            else:
                metadata = {}
            meshes[name] = {
                "vertices": v,
                "faces": f,
                "visual": visuals,
                "metadata": metadata,
                "process": False,
            }
            mesh_prim_replace[mesh_index] = [name]
        # avoid altering inside loop
        mesh_prim = mesh_prim_replace
        # remove outdated meshes
        [meshes.pop(p, None) for p in mesh_pop]

    # make it easier to reference nodes
    nodes = header.get("nodes", [])
    # nodes are referenced by index
    # save their string names if they have one
    # we have to accumulate in a for loop opposed
    # to a dict comprehension as it will be checking
    # the mutated dict in every loop
    name_index = {}
    name_counts = {}
    for i, n in enumerate(nodes):
        name_index[unique_name(n.get("name", str(i)), name_index, counts=name_counts)] = i
    # invert the dict so we can look up by index
    # node index (int) : name (str)
    names = {v: k for k, v in name_index.items()}

    # make sure we have a unique base frame name
    base_frame = "world"
    if base_frame in names:
        base_frame = str(int(np.random.random() * 1e10))
    names[base_frame] = base_frame

    # visited, kwargs for scene.graph.update
    graph = deque()
    # unvisited, pairs of node indexes
    queue = deque()

    # camera(s), if they exist
    camera = None
    camera_transform = None

    if "scene" in header:
        # specify the index of scenes if specified
        scene_index = header["scene"]
    else:
        # otherwise just use the first index
        scene_index = 0

    if "scenes" in header:
        # start the traversal from the base frame to the roots
        for root in header["scenes"][scene_index].get("nodes", []):
            # add transform from base frame to these root nodes
            queue.append((base_frame, root))

    # make sure we don't process an edge multiple times
    consumed = set()

    # go through the nodes tree to populate
    # kwargs for scene graph loader
    while len(queue) > 0:
        # (int, int) pair of node indexes
        edge = queue.pop()

        # avoid looping forever if someone specified
        # recursive nodes
        if edge in consumed:
            continue

        consumed.add(edge)
        a, b = edge

        # dict of child node
        # parent = nodes[a]
        child = nodes[b]
        # add edges of children to be processed
        if "children" in child:
            queue.extend([(b, i) for i in child["children"]])

        # kwargs to be passed to scene.graph.update
        kwargs = {"frame_from": names[a], "frame_to": names[b]}

        # grab matrix from child
        # parent -> child relationships have matrix stored in child
        # for the transform from parent to child
        if "matrix" in child:
            kwargs["matrix"] = (
                np.array(child["matrix"], dtype=np.float64).reshape((4, 4)).T
            )
        else:
            # if no matrix set identity
            kwargs["matrix"] = _EYE

        # Now apply keyword translations
        # GLTF applies these in order: T * R * S
        if "translation" in child:
            kwargs["matrix"] = np.dot(
                kwargs["matrix"], transformations.translation_matrix(child["translation"])
            )
        if "rotation" in child:
            # GLTF rotations are stored as (4,) XYZW unit quaternions
            # we need to re- order to our quaternion style, WXYZ
            quat = np.reshape(child["rotation"], 4)[[3, 0, 1, 2]]
            # add the rotation to the matrix
            kwargs["matrix"] = np.dot(
                kwargs["matrix"], transformations.quaternion_matrix(quat)
            )
        if "scale" in child:
            # add scale to the matrix
            kwargs["matrix"] = np.dot(
                kwargs["matrix"], np.diag(np.concatenate((child["scale"], [1.0])))
            )

        # If a camera exists, create the camera and dont add the node to the graph
        # TODO only process the first camera, ignore the rest
        # TODO assumes the camera node is child of the world frame
        # TODO will only read perspective camera
        if "camera" in child and camera is None:
            cam_idx = child["camera"]
            try:
                camera = _cam_from_gltf(header["cameras"][cam_idx])
            except KeyError:
                log.debug("GLTF camera is not fully-defined")
            if camera:
                camera_transform = kwargs["matrix"]
            continue

        # treat node metadata similarly to mesh metadata
        if isinstance(child.get("extras"), dict):
            kwargs["metadata"] = child["extras"]

        # put any node extensions in a field of the metadata
        if "extensions" in child:
            if "metadata" not in kwargs:
                kwargs["metadata"] = {}
            kwargs["metadata"]["gltf_extensions"] = child["extensions"]

        if "mesh" in child:
            geometries = mesh_prim[child["mesh"]]

            # if the node has a mesh associated with it
            if len(geometries) > 1:
                # append root node
                graph.append(kwargs.copy())
                # put primitives as children
                for geom_name in geometries:
                    # save the name of the geometry
                    kwargs["geometry"] = geom_name
                    # no transformations
                    kwargs["matrix"] = _EYE
                    kwargs["frame_from"] = names[b]
                    # if we have more than one primitive assign a new UUID
                    # frame name for the primitives after the first one
                    frame_to = f"{names[b]}_{util.unique_id(length=6)}"
                    kwargs["frame_to"] = frame_to
                    # append the edge with the mesh frame
                    graph.append(kwargs.copy())
            elif len(geometries) == 1:
                kwargs["geometry"] = geometries[0]
                if "name" in child:
                    kwargs["frame_to"] = names[b]
                graph.append(kwargs.copy())
        else:
            # if the node doesn't have any geometry just add
            graph.append(kwargs)

    # kwargs for load_kwargs
    result = {
        "class": "Scene",
        "geometry": meshes,
        "graph": graph,
        "base_frame": base_frame,
        "camera": camera,
        "camera_transform": camera_transform,
    }
    try:
        # load any scene extras into scene.metadata
        # use a try except to avoid nested key checks
        result["metadata"] = header["scenes"][header["scene"]]["extras"]
    except BaseException:
        pass
    try:
        # load any scene extensions into a field of scene.metadata
        # use a try except to avoid nested key checks
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["gltf_extensions"] = header["extensions"]
    except BaseException:
        pass

    return result


def _cam_from_gltf(cam):
    """
    Convert a gltf perspective camera to trimesh.

    The retrieved camera will have default resolution, since the gltf specification
    does not contain it.

    If the camera is not perspective will return None.
    If the camera is perspective but is missing fields, will raise `KeyError`

    Parameters
    ------------
    cam : dict
      Camera represented as a dictionary according to glTF

    Returns
    -------------
    camera : trimesh.scene.cameras.Camera
      Trimesh camera object
    """
    if "perspective" not in cam:
        return
    name = cam.get("name")
    znear = cam["perspective"]["znear"]
    aspect_ratio = cam["perspective"]["aspectRatio"]
    yfov = np.degrees(cam["perspective"]["yfov"])

    fov = (aspect_ratio * yfov, yfov)

    return Camera(name=name, fov=fov, z_near=znear)
