import taichi as ti
import utils
from fluid_simulator import *

@ti.data_oriented
class Initializer2D:
    def __init__(self, res, x0, y0, x1, y1, blocks):
        self.res = res
        self.x0 = int(res * x0)
        self.y0 = int(res * y0)
        self.x1 = int(res * x1)
        self.y1 = int(res * y1)
        self.blocks = blocks

    @ti.kernel
    def init_kernel(self, cell_type : ti.template()):
        for i, j in cell_type:
            in_block = False
            for k in ti.static(range(len(self.blocks))):
                if i >= self.blocks[k][0] * self.res and i <= self.blocks[k][2] * self.res and \
                   j >= self.blocks[k][1] * self.res and j <= self.blocks[k][3] * self.res:
                    in_block = True
            if in_block:
                cell_type[i, j] = utils.SOLID
            elif i >= self.x0 and i <= self.x1 and j >= self.y0 and j <= self.y1:
                cell_type[i, j] = utils.FLUID

    def init_scene(self, simulator):
        simulator.level_set.initialize_with_aabb((self.x0 * simulator.dx, self.y0 * simulator.dx), (self.x1 * simulator.dx, self.y1 * simulator.dx))
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
        
