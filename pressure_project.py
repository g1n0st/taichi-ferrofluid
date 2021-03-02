import taichi as ti
from mgpcg import MGPCGPoissonSolver
import utils

@ti.data_oriented
class PressureProjectStrategy:
    def __init__(self, dim, velocity, ghost_fluid_method, phi, p0):
        self.dim = dim

        self.velocity = velocity
        self.ghost_fluid_method = ghost_fluid_method
        self.phi = phi
        self.p0 = p0 # the standard atmospheric pressure

    @ti.kernel
    def build_b_kernel(self, 
                       cell_type : ti.template(), 
                       b : ti.template()):
        for I in ti.grouped(cell_type):
            if cell_type[I] == utils.FLUID:
                for k in ti.static(range(self.dim)):
                    offset = ti.Vector.unit(self.dim, k)
                    b[I] += (self.velocity[k][I] - self.velocity[k][I + offset])
                b[I] *= self.scale_b

        for I in ti.grouped(cell_type):
            if cell_type[I] == utils.FLUID:
                for k in ti.static(range(self.dim)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(self.dim, k) * s
                        if cell_type[I + offset] == utils.SOLID:
                            if s < 0: b[I] -= self.scale_b * (self.velocity[k][I] - 0)
                            else: b[I] += self.scale_b * (self.velocity[k][I + offset] - 0)
                        elif cell_type[I + offset] == utils.AIR:
                            if ti.static(self.ghost_fluid_method):
                                b[I] += self.scale_A * self.p0
                            else:
                                c = (self.phi[I] - self.phi[I + offset]) / self.phi[I]
                                c = utils.clamp(c, -1e3, 1e3)
                                b[I] += self.scale_A * c * self.p0

    def build_b(self, solver : MGPCGPoissonSolver):
        self.build_b_kernel(solver.grid_type[0], 
                            solver.b)

    @ti.kernel
    def build_A_kernel(self, 
                       level : ti.template(),
                       grid_type : ti.template(), 
                       Adiag : ti.template(), 
                       Ax : ti.template()):
        for I in ti.grouped(grid_type):
            if grid_type[I] == utils.FLUID:
                for k in ti.static(range(self.dim)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(self.dim, k) * s
                        if grid_type[I + offset] == utils.FLUID: 
                            Adiag[I] += self.scale_A
                            if ti.static(s > 0): Ax[I][k] = -self.scale_A
                        elif grid_type[I + offset] == utils.AIR:
                            if ti.static(self.ghost_fluid_method and level == 0):
                                c = (self.phi[I] - self.phi[I + offset]) / self.phi[I]
                                c = utils.clamp(c, -1e3, 1e3)
                                Adiag[I] += self.scale_A * c
                            else:
                                Adiag[I] += self.scale_A

    def build_A(self, solver : MGPCGPoissonSolver, level):
        self.build_A_kernel(level, 
                            solver.grid_type[level], 
                            solver.Adiag[level], 
                            solver.Ax[level])

