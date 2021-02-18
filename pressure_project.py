import taichi as ti
from mgpcg import MGPCGPoissonSolver
import utils

@ti.data_oriented
class PressureProjectStrategy:
    def __init__(self, velocity):
        self.scale_A = 1.0
        self.scale_b = 1.0
        self.velocity = velocity

    @ti.kernel
    def build_b_kernel(self, res : ti.template(), dim : ti.template(), cell_type : ti.template(), b : ti.template()):
        for I in ti.grouped(cell_type):
            if cell_type[I] == utils.FLUID:
                for k in ti.static(range(dim)):
                    offset = ti.Vector.unit(dim, k)
                    b[I] += (self.velocity[k][I] - self.velocity[k][I + offset])
                b[I] *= self.scale_b

        for I in ti.grouped(cell_type):
            if cell_type[I] == utils.FLUID:
                for k in ti.static(range(dim)):
                    offset = ti.Vector.unit(dim, k)
                    if I[k] == 0 or cell_type[I - offset] == utils.SOLID:
                        b[I] -= self.scale_b * (self.velocity[k][I] - 0)
                    if I[k] == res[k] - 1 or cell_type[I + offset] == utils.SOLID:
                        b[I] += self.scale_b * (self.velocity[k][I + offset] - 0)

    def build_b(self, solver : MGPCGPoissonSolver):
        self.build_b_kernel(solver.res, solver.dim, solver.grid_type[0], solver.b)
   
    @ti.kernel
    def build_A_kernel(self, dim : ti.template(), l : ti.template(), grid_type : ti.template(), Adiag : ti.template(), Ax : ti.template()):
        scale = self.scale_A

        for I in ti.grouped(grid_type):
            if grid_type[I] == utils.FLUID:
                for k in ti.static(range(dim)):
                    offset = ti.Vector.unit(dim, k)
                    if grid_type[I - offset] == utils.FLUID: 
                        Adiag[I] += scale
                    elif grid_type[I - offset] == utils.AIR:
                        Adiag[I] += scale
                    if grid_type[I + offset] == utils.FLUID: 
                        Adiag[I] += scale
                        Ax[I][k] = -scale
                    elif grid_type[I + offset] == utils.AIR:
                        Adiag[I] += scale

    def build_A(self, solver : MGPCGPoissonSolver, level):
        self.build_A_kernel(solver.dim, level, solver.grid_type[level], solver.Adiag[level], solver.Ax[level])
