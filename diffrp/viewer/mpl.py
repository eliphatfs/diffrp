"""
Experimental. Implementation and interface subject to change.
"""
import time
import numpy
import torch
import torch_redstone as rst
from matplotlib import animation
import matplotlib.pyplot as plotlib
from ..rendering.camera import PerspectiveCamera
from ..utils import background_alpha_compose, agx_base_contrast
from ..rendering.surface_deferred import SurfaceDeferredRenderSession, SurfaceDeferredRenderSessionOptions


class Viewer:
    """
    View a scene by constructing the viewer, with the initial camera placed with the specified pose.
    Requires ``matplotlib`` to be installed.
    
    If you need semi-transparency you can tune up max_layers to the overdraw you want to support,
    but the performance would be lower.
    
    Use WASDQE to navigate around the scene, and rotate the view by holding the right mouse button.
    """
    def __init__(self, scene, init_azim: int = 0, init_elev: int = 0, init_origin: list = [0, 0, 0], max_layers: int = 1):
        for key in plotlib.rcParams:
            if key.startswith('keymap.'):
                plotlib.rcParams[key] = []
        self.scene = scene

        self.azim = init_azim
        self.elev = init_elev
        self.origin = numpy.array(init_origin.copy(), dtype=numpy.float32)
        self.key_state = set()
        self.sensitivity = 1.0
        self.max_layers = max_layers
        self.rebuild_camera()
        torch.set_grad_enabled(False)

        self.r_drag_pos = None
        self.ibl_cache = None
        self.last_update = None

        fig = plotlib.figure()
        ax: plotlib.Axes = plotlib.axes(navigate=False)
        ax.axis('off')
        self.widget = ax.imshow(rst.torch_to_numpy(self.render()))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        fig.canvas.mpl_connect('key_press_event', self.on_key_down)
        fig.canvas.mpl_connect('key_release_event', self.on_key_up)
        self.anim = animation.FuncAnimation(fig, self.refresh, interval=10, cache_frame_data=False, blit=True)
        plotlib.show()

    def update(self):
        t = time.perf_counter()
        if self.last_update is None:
            self.last_update = t
        delta = t - self.last_update
        velocity = numpy.zeros(3, dtype=numpy.float32)
        if 'a' in self.key_state:
            velocity[0] -= 1
        if 'd' in self.key_state:
            velocity[0] += 1
        if 'w' in self.key_state:
            velocity[2] -= 1
        if 's' in self.key_state:
            velocity[2] += 1
        if 'q' in self.key_state:
            velocity[1] -= 1
        if 'e' in self.key_state:
            velocity[1] += 1
        self.origin += velocity * delta * 5
        self.rebuild_camera()
        self.last_update = t

    def render(self):
        self.update()
        rp = SurfaceDeferredRenderSession(
            self.scene, self.camera, opaque_only=False,
            options=SurfaceDeferredRenderSessionOptions(max_layers=3)
        )
        if self.ibl_cache is None:
            self.ibl_cache = rp.prepare_ibl()
        else:
            rp.set_prepare_ibl(self.ibl_cache)
        rgb = background_alpha_compose(1, rp.albedo())
        return agx_base_contrast(rgb)

    def refresh(self, frame):
        import time
        a = time.perf_counter()
        render = self.render()
        b = time.perf_counter()
        render = rst.torch_to_numpy(render)
        c = time.perf_counter()
        self.widget.set_data(render)
        d = time.perf_counter()
        print("Render %.1f ms, Read %.1f ms, Set %.1f ms" % (1000 * (b - a), 1000 * (c - b), 1000 * (d - c)))
        return (self.widget,)

    def rebuild_camera(self):
        self.camera = PerspectiveCamera.from_orbit(512, 512, 3.0, self.azim, self.elev, self.origin, near=0.01, far=50.0)
    
    def on_key_down(self, event):
        self.key_state.add(event.key)

    def on_key_up(self, event):
        try:
            self.key_state.remove(event.key)
        except KeyError:
            import traceback; traceback.print_exc()

    def on_press(self, event):
        if event.button == 3:
            self.r_drag_pos = event.xdata, event.ydata

    def on_motion(self, event):
        if self.r_drag_pos is not None:
            oldx, oldy = self.r_drag_pos
            self.r_drag_pos = event.xdata, event.ydata
            self.azim -= event.xdata - oldx
            self.elev += event.ydata - oldy
            self.elev = min(90, max(self.elev, -90))

    def on_release(self, event):
        if event.button == 3:
            self.r_drag_pos = None
