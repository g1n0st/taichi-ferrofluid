import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, res, mode, output):
        self.grid_res = res
        self.res = 512
        self.mode = mode
        self.gui = ti.GUI("demo", (self.res, self.res))
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.output = output
        self.frame = 0
    
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
            x = int(fx)
            y = int(fy)

            if cell_type[x, y] == utils.SOLID: self.color_buffer[i, j] = ti.Vector([0, 0, 0])
            elif utils.sample(phi, ti.Vector([fx, fy])) <= 0: self.color_buffer[i, j] = ti.Vector([113 / 255, 131 / 255, 247 / 255]) # fluid
            else: self.color_buffer[i, j] = ti.Vector([0.99, 0.99, 0.99])

    def set_color_buffer(self):
        img = self.color_buffer.to_numpy()
        self.gui.set_image(img)

    def visualize_pressure(self, simulator):
        self.fill_pressure(simulator.res, simulator.pressure)
        self.set_color_buffer()
        
    def visualize_levelset(self, simulator):
        self.fill_levelset(simulator.res, simulator.level_set.phi, simulator.dx)
        self.set_color_buffer()
        
    def visualize_normal(self, simulator):
        self.fill_normal(simulator.res, simulator.surface_tension.n)
        self.set_color_buffer()

    def visualize_psi(self, simulator):
        self.fill_psi(simulator.res, simulator.psi, 50000)
        self.set_color_buffer()

    def visual(self, simulator):
        self.visualize_kernel(simulator.res, simulator.level_set.phi, simulator.cell_type)
        self.set_color_buffer()

        self.gui.text(f'time = {simulator.total_t:.3f}s', [0.3, 0.95], font_size = 32, color = 0x0)
            
    def visualize_particles(self, simulator):
        bg_color = 0x000000
        particle_color = 0x0FFFFF
        particle_radius = 1.0
        pos = simulator.p_x.to_numpy()
        
        self.gui.clear(bg_color)
        self.gui.circles(pos / (self.grid_res * simulator.dx), radius=particle_radius, color=particle_color)

    def visualize(self, simulator):
        if self.mode == 'pressure':
            self.visualize_pressure(simulator)
        elif self.mode == 'particles':
            self.visualize_particles(simulator)
        elif self.mode == 'levelset':
            self.visualize_levelset(simulator)
        elif self.mode == 'normal':
            self.visualize_normal(simulator)
        elif self.mode == 'visual':
            self.visual(simulator)
        elif self.mode == 'psi':
            self.visualize_psi(simulator)

        if self.output:
            self.gui.show(f'{self.frame:06d}.png')
        else:
            self.gui.show()

        self.frame += 1

