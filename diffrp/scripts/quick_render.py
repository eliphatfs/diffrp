import os
import argparse
import diffrp
import psutil
import trimesh
import rich.progress
from diffrp.loaders.gltf_loader import _load_glb
from diffrp.utils import *
from diffrp.resources.hdris import newport_loft
from multiprocessing.pool import ThreadPool


@torch.no_grad()
def normalize_to_2_unit_cube(scene: diffrp.Scene):
    # Get the bounding box of the mesh
    bmin = diffrp.gpu_f32([1e30] * 3)
    bmax = diffrp.gpu_f32([-1e30] * 3)
    world_v = [diffrp.transform_point4x3(prim.verts, prim.M) for prim in scene.objects]
    for verts, prim in zip(world_v, scene.objects):
        bmin = torch.minimum(bmin, verts.min(0)[0])
        bmax = torch.maximum(bmax, verts.max(0)[0])
    min_bounds = bmin
    max_bounds = bmax
    
    # Compute the center of the mesh
    center = ((min_bounds + max_bounds) / 2.0).cpu().numpy()
    
    # Compute the maximum dimension of the bounding box
    scale = torch.max(max_bounds - min_bounds).item()
    
    T = trimesh.transformations.translation_matrix(-center)
    S = trimesh.transformations.scale_matrix(2.0 / scale)
    M = diffrp.gpu_f32(S @ T)
    for prim in scene.objects:
        prim.M = M @ prim.M
    return scene


def load(p: str):
    if p.endswith(".glb"):
        with open(p, "rb") as fi:
            return os.path.basename(p), trimesh.load(_load_glb(fi), force='scene', process=False)
    else:
        return os.path.basename(p), trimesh.load(p, force='scene', process=False)


def build_argparse(argp: argparse.ArgumentParser):
    argp.set_defaults(entrypoint=main)
    argp.add_argument("-i", "--glb-path", required=True)
    argp.add_argument("-o", "--output-path", default='rendered')
    argp.add_argument("-t", "--io-threads", type=int, default=psutil.cpu_count(False))
    argp.add_argument("-r", "--resolution", type=int, default=512)
    argp.add_argument("-s", "--ssaa", type=int, default=2)
    argp.add_argument("-f", "--fov", type=int, default=30)
    argp.add_argument("-a", "--azim-list", type=str, default='0,45,90,135,180,225,270,315')
    argp.add_argument("-e", "--elev-list", type=str, default='0,30,-15')


@torch.no_grad()
def main(args):
    todo = []
    
    if os.path.isfile(args.glb_path):
        todo.append(args.glb_path)
    else:
        for r, ds, fs in os.walk(args.glb_path):
            for f in fs:
                if f.endswith('.glb'):
                    todo.append(os.path.join(r, f))
    write_jobs = []
    light_cache = None
    os.makedirs(args.output_path, exist_ok=True)
    image_per_obj = len(args.elev_list.split(",")) * len(args.azim_list.split(","))
    with ThreadPool(args.io_threads) as pool:
        with rich.progress.Progress() as prog:
            obj_task = prog.add_task("Rendering...", total=len(todo))
            sub_task = prog.add_task("", total=image_per_obj)
            for name, scene in pool.imap_unordered(load, todo):
                prog.update(obj_task, advance=1)
                prog.update(sub_task, total=image_per_obj, completed=0, description=name)
                scene = diffrp.from_trimesh_scene(scene, compute_tangents=True).static_batching()
                scene = normalize_to_2_unit_cube(scene)
                cam_id = 0
                scene.add_light(diffrp.ImageEnvironmentLight(1.0, gpu_f32([1.0] * 3), newport_loft().cuda(), False))
                for elev in map(float, args.elev_list.split(",")):
                    for azim in map(float, args.azim_list.split(",")):
                        R = args.resolution
                        A = args.ssaa
                        F = args.fov
                        radius = 1.5 / math.sin(math.radians(F / 2))
                        camera = diffrp.PerspectiveCamera.from_orbit(R * A, R * A, radius, azim, elev, [0, 0, 0], F, far=20)
                        cam_id += 1
                        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
                        if light_cache is None:
                            light_cache = rp.prepare_ibl()
                        else:
                            rp.set_prepare_ibl(light_cache)
                        pbr_premult = rp.compose_layers(
                            rp.pbr_layered() + [torch.zeros([R * A, R * A, 3], device='cuda')],
                            rp.alpha_layered() + [torch.zeros([R * A, R * A, 1], device='cuda')]
                        )
                        pbr = ssaa_downscale(pbr_premult, 2)
                        pbr = float4(agx_base_contrast(pbr.rgb / torch.clamp_min(pbr.a, 0.0001)), pbr.a)
                        prog.update(sub_task, advance=1)
                        write_jobs.append(pool.apply_async(
                            to_pil(pbr).save,
                            (os.path.join(args.output_path, f"{cam_id:02}_{name}.png"),),
                            dict(compress_level=1)
                        ))
        for job in rich.progress.track(write_jobs, "Writing..."):
            job.get()


if __name__ == '__main__':
    main()
