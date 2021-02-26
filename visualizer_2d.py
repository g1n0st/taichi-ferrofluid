import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, res, mode):
        self.grid_res = res
        self.res = 512
        self.mode = mode
        self.gui = ti.GUI("demo", (self.res, self.res))
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
    
    @ti.kernel
    def fill_pressure(self, grid_res : ti.template(), p : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * grid_res[0])
            y = int((j + 0.5) / self.res * grid_res[1])

            m = (ti.log(min(p[x, y], 1000000) + 1) / ti.log(10)) / 6.001
            self.color_buffer[i, j] = ti.Vector([m, m, m])

    @ti.kernel
    def fill_levelset(self, grid_res : ti.template(), phi : ti.template(), dx : ti.template(), valid : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * grid_res[0])
            y = int((j + 0.5) / self.res * grid_res[1])

            p = min(phi[x, y] / (dx * grid_res[0]) * 100, 1)

            # if valid[x, y] == 1: self.color_buffer[i, j] = ti.Vector([1, 1, 1])
            if p > 0: self.color_buffer[i, j] = ti.Vector([p, 0, 0])
            else: self.color_buffer[i, j] = ti.Vector([0, 0, -p])

    def visualize_pressure(self, simulator):
        self.fill_pressure(simulator.res, simulator.pressure)
        img = self.color_buffer.to_numpy()
        self.gui.set_image(img)
        self.gui.show()

    def visualize_levelset(self, simulator):
        self.fill_levelset(simulator.res, simulator.level_set.phi, simulator.dx, simulator.level_set.valid)
        img = self.color_buffer.to_numpy()
        self.gui.set_image(img)
        self.gui.show()
    
    def visualize_particles(self, simulator):
        bg_color = 0x000000
        particle_color = 0x0FFFFF
        particle_radius = 1.0
        pos = simulator.p_x.to_numpy()
        
        self.gui.clear(bg_color)
        self.gui.circles(pos / (self.grid_res * simulator.dx), radius=particle_radius, color=particle_color)
        self.gui.show()

    def visualize(self, simulator):
        if self.mode == 'pressure':
            self.visualize_pressure(simulator)
        elif self.mode == 'particles':
            self.visualize_particles(simulator)
        elif self.mode == 'levelset':
            self.visualize_levelset(simulator)

