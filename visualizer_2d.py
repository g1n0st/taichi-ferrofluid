import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer2D:
    def __init__(self, res, switch = 1):
        self.grid_res = res
        self.res = 512
        self.switch = switch
        self.gui = ti.GUI("demo", (self.res, self.res))
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
    
    @ti.kernel
    def fill_marker(self, dx : ti.template(), p : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * dx)
            y = int((j + 0.5) / self.res * dx)

            m = (ti.log(min(p[x, y], 100) + 1) / ti.log(10)) / 2.001
            self.color_buffer[i, j] = ti.Vector([m, m, m])

    def visualize_1(self, simulator):
        self.fill_marker(simulator.dx * 128, simulator.pressure)
        img = self.color_buffer.to_numpy()
        self.gui.set_image(img)
        self.gui.show()
    
    def visualize_2(self, simulator):
        bg_color = 0x000000
        particle_color = 0x0FFFFF
        particle_radius = 1.0
        pos = simulator.markers.to_numpy()
        
        self.gui.clear(bg_color)
        self.gui.circles(pos / (self.grid_res * simulator.dx), radius=particle_radius, color=particle_color)
        self.gui.show()

    def visualize(self, simulator):
        if self.switch == 1:
            self.visualize_1(simulator)
        else:
            self.visualize_2(simulator)

