import taichi as ti
import utils
from fluid_simulator import *

@ti.data_oriented
class Initializer3D: # tmp initializer
    def __init__(self, res, x0, y0, z0, x1, y1, z1):
        self.res = res
        self.x0 = int(res * x0)
        self.y0 = int(res * y0)
        self.z0 = int(res * z0)
        self.x1 = int(res * x1)
        self.y1 = int(res * y1)
        self.z1 = int(res * z1)


    @ti.kernel
    def init_kernel(self, cell_type : ti.template()):
        for i, j, k in cell_type:
            if i >= self.x0 and i <= self.x1 and \
               j >= self.y0 and j <= self.y1 and \
               k >= self.z0 and k <= self.z1:
                cell_type[i, j, k] = utils.FLUID

    def init_scene(self, simulator):
        self.init_kernel(simulator.cell_type)

