import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, grid_res, res):
        self.grid_res = grid_res
        self.res = res
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
    
    @ti.kernel
    def fill_pressure(self, grid_res : ti.template(), p : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * grid_res[0])
            y = int((j + 0.5) / self.res * grid_res[1])

            m = (ti.log(min(p[x, y], 1000000) + 1) / ti.log(10)) / 6.001
            self.color_buffer[i, j] = ti.Vector([m, m, m])

    @ti.kernel
    def fill_levelset(self, grid_res : ti.template(), phi : ti.template(), dx : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * grid_res[0])
            y = int((j + 0.5) / self.res * grid_res[1])

            p = min(phi[x, y] / (dx * grid_res[0]) * 100, 1)

            if p > 0: self.color_buffer[i, j] = ti.Vector([p, 0, 0])
            else: self.color_buffer[i, j] = ti.Vector([0, 0, -p])

    @ti.kernel
    def fill_normal(self, grid_res : ti.template(), n : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * grid_res[0])
            y = int((j + 0.5) / self.res * grid_res[1])

            r = (n[x, y][0] + 1) * 0.5
            g = (n[x, y][1] + 1) * 0.5
            self.color_buffer[i, j] = ti.Vector([r, g, 0])

    @ti.kernel
    def fill_psi(self, grid_res : ti.template(), psi : ti.template(), mx : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * grid_res[0])
            y = int((j + 0.5) / self.res * grid_res[1])

            p = psi[x, y] / mx
            self.color_buffer[i, j] = ti.Vector([p, 0, 0])

    @ti.kernel
    def visualize_kernel(self, grid_res : ti.template(), phi : ti.template(), cell_type : ti.template()):
        for i, j in self.color_buffer:
            fx = (i + 0.5) / self.res * grid_res[0]
            fy = (j + 0.5) / self.res * grid_res[1]
            x, y = int(fx), int(fy)

            if cell_type[x, y] == utils.SOLID: self.color_buffer[i, j] = ti.Vector([0, 0, 0])
            elif utils.sample(phi, ti.Vector([fx, fy])) <= 0: self.color_buffer[i, j] = ti.Vector([113 / 255, 131 / 255, 247 / 255]) # fluid
            else: self.color_buffer[i, j] = ti.Vector([0.99, 0.99, 0.99])

    def visualize_factory(self, simulator):
        if self.mode == 'pressure':
            self.fill_pressure(simulator.res, simulator.pressure)
        elif self.mode == 'levelset':
            self.fill_levelset(simulator.res, simulator.level_set.phi, simulator.dx)
        elif self.mode == 'normal':
            self.fill_normal(simulator.res, simulator.surface_tension.n)
        elif self.mode == 'visual':
            self.visualize_kernel(simulator.res, simulator.level_set.phi, simulator.cell_type)
        elif self.mode == 'psi':
            self.fill_psi(simulator.res, simulator.psi, 50000)

    def visualize(self, simulator):
        assert 0, 'Please use GUIVisualizer2D/VideoVisualizer2D'

@ti.data_oriented
class GUIVisualizer2D(Visualizer2D):
    def __init__(self, grid_res, res, mode, output = False, title = 'demo'):
        super().__init__(grid_res, res)

        self.mode = mode
        self.gui = ti.GUI(title, (self.res, self.res))
        self.output = output
        self.frame = 0
    
    def visualize(self, simulator):
        self.visualize_factory(simulator)

        img = self.color_buffer.to_numpy()
        self.gui.set_image(img)
        # self.gui.text(f'time = {simulator.total_t:.3f}s', [0.3, 0.95], font_size = 32, color = 0x0)

        if self.output:
            self.gui.show(f'{self.frame:06d}.png')
        else:
            self.gui.show()

        self.frame += 1

@ti.data_oriented
class VideoVisualizer2D(Visualizer2D):
    def __init__(self, grid_res, res, mode, result_dir = './results', video_rate = 24, auto = True):
        super().__init__(grid_res, res)

        self.mode = mode
        self.video_manager = ti.VideoManager(output_dir = result_dir,
                                        framerate = video_rate,
                                        automatic_build = auto)

        self.pixels = ti.field(ti.u8, shape=(res, res, 3))
   
    @ti.kernel
    def buffer_to_pixels(self):
        for i, j in self.color_buffer:
            for k in ti.static(range(3)):
                self.pixels[i, j, k] = int(self.color_buffer[i, j][k] * 255)

    def visualize(self, simulator):
        self.visualize_factory(simulator)
        self.buffer_to_pixels()

        img = self.pixels.to_numpy()
        self.video_manager.write_frame(img)
