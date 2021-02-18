import taichi as ti
import utils
from fluid_simulator import *

@ti.data_oriented
class Initializer3D: # tmp initializer
    def __init__(self, res, x, y, z):
        self.res = res
        self.xn = int(res * x)
        self.yn = int(res * y)
        self.zn = int(res * z)

    @ti.kernel
    def init_kernel(self, cell_type : ti.template()):
        for i, j, k in cell_type:
            if i <= self.xn and j >= self.yn and k <= self.zn:
                cell_type[i, j, k] = utils.FLUID

    def init_scene(self, simulator):
        self.init_kernel(simulator.cell_type)

