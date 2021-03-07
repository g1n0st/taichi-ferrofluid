import taichi as ti
import utils
from fluid_simulator import *

@ti.data_oriented
class Initializer2D:
    def __init__(self, res, x0, y0, x, y, x1, x2, y1, y2):
        self.res = res
        self.x0 = int(res * x0)
        self.y0 = int(res * y0)
        self.xn = int(res * x)
        self.yn = int(res * y)
        self.x1 = int(res * x1)
        self.x2 = int(res * x2)
        self.y1 = int(res * y1)
        self.y2 = int(res * y2)

    @ti.kernel
    def init_kernel(self, cell_type : ti.template()):
        for i, j in cell_type:
            if i >= self.x0 and i <= self.xn and j >= self.y0 and j <= self.yn:
                cell_type[i, j] = utils.FLUID
            elif i >= self.x1 and i <= self.x2 and j >= self.y1 and j <= self.y2:
                cell_type[i, j] = utils.SOLID

    def init_scene(self, simulator):
        simulator.level_set.initialize_with_aabb((self.x0 * simulator.dx, self.y0 * simulator.dx), (self.xn * simulator.dx, self.yn * simulator.dx))
        self.init_kernel(simulator.cell_type)

@ti.data_oriented
class SphereInitializer2D:
    def __init__(self, res, x0, y0, r):
        self.res = res
        self.x0 = int(res * x0)
        self.y0 = int(res * y0)
        self.r = int(res * r)

    @ti.kernel
    def init_kernel(self, cell_type : ti.template()):
        for i, j in cell_type:
            if (i - self.x0) ** 2 + (j - self.y0) ** 2 <= self.r ** 2:
                cell_type[i, j] = utils.FLUID

    def init_scene(self, simulator):
        simulator.level_set.initialize_with_sphere((self.x0 * simulator.dx, self.y0 * simulator.dx), self.r * simulator.dx)
        self.init_kernel(simulator.cell_type)
        
