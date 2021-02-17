import taichi as ti
from mgpcg import MGPCGPoissonSolver
import utils

@ti.data_oriented
class PressureProjectStrategy:
    def __init__(self, scale_A, scale_b, velocity):
        self.scale_A = scale_A
        self.scale_b = scale_b
        self.velocity = velocity

    @ti.kernel
    def init_b(self, res : ti.template(), dim : ti.template(), cell_type : ti.template(), b : ti.template()):
        for I in ti.grouped(cell_type):
            if cell_type[I] == utils.FLUID:
                for k in ti.static(range(dim)):
                    offset = ti.Vector.unit(dim, k)
                    b[I] += (self.velocity[k][I + offset] - self.velocity[k][I])
                b[I] *= -1 * self.scale_b

        for I in ti.grouped(cell_type):
            if cell_type[I] == utils.FLUID:
                for k in ti.static(range(dim)):
                    offset = ti.Vector.unit(dim, k)
                    if I[k] == 0 or cell_type[I - offset] == utils.SOLID:
                        b[I] -= self.scale_b * (self.velocity[k][I] - 0)
                    if I[k] == res[k] - 1 or cell_type[I + offset] == utils.SOLID:
                        b[I] += self.scale_b * (self.velocity[k][I + offset] - 0)

    def build_rhs(self, solver : MGPCGPoissonSolver):
        self.init_b(solver.res, solver.dim, solver.grid_type[0], solver.b)
   
    @ti.kernel
    def preconditioner_init(self, dim : ti.template(), l : ti.template(), grid_type : ti.template(), Adiag : ti.template(), Ax : ti.template()):
        scale = self.scale_A

        for I in ti.grouped(grid_type):
            if grid_type[I] == utils.FLUID:
                for k in ti.static(range(dim)):
                    offset = ti.Vector.unit(dim, k)
                    if grid_type[I - offset] == utils.FLUID: 
                        Adiag[I] += scale
                    elif grid_type[I - offset] == utils.AIR:
                        Adiag[I] += scale
                        Ax[I - offset][k] = -scale
                    if grid_type[I + offset] == utils.FLUID: 
                        Adiag[I] += scale
                        Ax[I][k] = -scale
                    elif grid_type[I + offset] == utils.AIR:
                        Adiag[I] += scale

    def build_lhs(self, solver : MGPCGPoissonSolver, level):
        self.preconditioner_init(solver.dim, level, solver.grid_type[level], solver.Adiag[level], solver.Ax[level])
