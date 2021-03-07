import taichi as ti
import taichi_glsl as ts
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, grid_res, res):
        self.grid_res = grid_res
        self.res = res
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))

    @ti.func
    def ij_to_xy(self, i, j):
        return int((i + 0.5) / self.res * self.grid_res), \
               int((j + 0.5) / self.res * self.grid_res)

    @ti.kernel
    def fill_pressure(self, p : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            m = (ti.log(min(p[x, y], 1e6) + 1) / ti.log(10)) / 6
            self.color_buffer[i, j] = ti.Vector([m, m, m])

    @ti.kernel
    def fill_levelset(self, phi : ti.template(), dx : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            p = min(phi[x, y] / (dx * self.grid_res) * 1e2, 1)
            if p > 0: self.color_buffer[i, j] = ti.Vector([p, 0, 0])
            else: self.color_buffer[i, j] = ti.Vector([0, 0, -p])

    @ti.kernel
    def fill_normal(self, n : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            r = (n[x, y][0] + 1) * 0.5
            g = (n[x, y][1] + 1) * 0.5
            self.color_buffer[i, j] = ti.Vector([r, g, 0])

    @ti.kernel
    def fill_psi(self, psi : ti.template(), min_psi : ti.f32, max_psi : ti.f32):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            p = (psi[x, y] - min_psi) / (max_psi - min_psi) # mapping to [0, 1]
            self.color_buffer[i, j] = ti.Vector([p, p, p])

    @ti.kernel
    def visualize_kernel(self, phi : ti.template(), cell_type : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            if cell_type[x, y] == utils.SOLID: 
                self.color_buffer[i, j] = ti.Vector([0, 0, 0])
            elif phi[x, y] <= 0: 
                self.color_buffer[i, j] = ti.Vector([113 / 255, 131 / 255, 247 / 255]) # fluid
            else: 
                self.color_buffer[i, j] = ti.Vector([1, 1, 1])

    def visualize_factory(self, simulator):
        if self.mode == 'pressure':
            self.fill_pressure(simulator.pressure)
        elif self.mode == 'levelset':
            self.fill_levelset(simulator.level_set.phi, simulator.dx)
        elif self.mode == 'normal':
            self.fill_normal(simulator.surface_tension.n)
        elif self.mode == 'visual':
            self.visualize_kernel(simulator.level_set.phi, simulator.cell_type)
        elif self.mode == 'psi':
            psi = simulator.psi.to_numpy()
            max_psi, min_psi = float(np.max(psi)), float(np.min(psi))
            self.fill_psi(simulator.psi, min_psi, max_psi)

    def visualize(self, simulator):
        assert 0, 'Please use GUIVisualizer2D/VideoVisualizer2D'

@ti.data_oriented
class GUIVisualizer2D(Visualizer2D):
    def __init__(self, grid_res, res, mode, output = False, text = False, title = 'demo'):
        super().__init__(grid_res, res)

        self.mode = mode
        self.gui = ti.GUI(title, (self.res, self.res))
        self.output = output
        self.frame = 0
        self.text = text

        self.angle_x = ti.field(dtype=ti.f32, shape=(self.res, self.res)) # H_x
        self.angle_y = ti.field(dtype=ti.f32, shape=(self.res, self.res)) # H_y

    @ti.kernel
    def fill_H(self, Hx : ti.template(), Hy : ti.template()):
        for i, j in self.color_buffer:
            x, y = self.ij_to_xy(i, j)

            if ti.abs(Hx[x, y]) < 1e-4 and ti.abs(Hy[x, y]) < 1e-4:
                self.angle_x[x, y] = 0
                self.angle_y[x, y] = 0
            else:
                H = ts.normalize(ti.Vector([Hx[x, y], Hy[x, y]]))
                self.angle_x[x, y] = H.x
                self.angle_y[x, y] = H.y

    def visualize(self, simulator):
        if self.mode == 'H':
            self.fill_H(simulator.H[0], simulator.H[1])
            angle_x = self.angle_x.to_numpy()
            angle_y = self.angle_y.to_numpy()
            unit = 0.35 / self.grid_res
            for i in range(self.grid_res):
                for j in range(self.grid_res):
                    o = np.asarray([(i + 0.5) / self.grid_res, (j + 0.5) / self.grid_res])
                    d = np.asarray([angle_x[i, j], angle_y[i, j]])
                    self.gui.line(begin = o, end = o + d * unit, color = 0xFF0000, radius = 1.5)
                    self.gui.line(begin = o, end = o - d * unit, color = 0x0000FF, radius = 1.5)
        else:
            self.visualize_factory(simulator)
            img = self.color_buffer.to_numpy()
            self.gui.set_image(img)
            if self.text:
                self.gui.text(f'time = {simulator.total_t:.3f}s', [0.3, 0.95], font_size = 32, color = 0x0)

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
