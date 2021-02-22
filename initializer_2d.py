import taichi as ti
import utils
from fluid_simulator import *

@ti.data_oriented
class Initializer2D:
    def __init__(self, res, x, y, x1, x2, y1, y2):
        self.res = res
        self.xn = int(res * x)
        self.yn = int(res * y)
        self.x1 = int(res * x1)
        self.x2 = int(res * x2)
        self.y1 = int(res * y1)
        self.y2 = int(res * y2)

    @ti.kernel
    def init_kernel(self, cell_type : ti.template()):
        for i, j in cell_type:
            if i <= self.xn and j >= self.yn:
                cell_type[i, j] = utils.FLUID
            elif i >= self.x1 and i <= self.x2 and j >= self.y1 and j <= self.y2:
                cell_type[i, j] = utils.SOLID

    def init_scene(self, simulator):
        simulator.level_set.initialize_with_aabb((0, self.yn * simulator.dx), (self.xn * simulator.dx, self.res * simulator.dx))
        self.init_kernel(simulator.cell_type)

