import taichi as ti
from mgpcg import MGPCGPoissonSolver
import utils

@ti.data_oriented
class PotentialFunctionStrategy:
    def __init__(self, dim, res, dx, H_ext, chi):
        self.dim = dim
        self.res = res
        self.dx = dx

        self.H_ext = H_ext
        self.chi = chi

    @ti.kernel
    def build_b_kernel(self,
                       cell_type : ti.template(),
                       b : ti.template()):
        for I in ti.grouped(cell_type): # solve the entire domain tao
            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    if I[k] + s >= 0 and I[k] + s < self.res[k]: # magnetic shielding
                        b[I] += utils.sample(self.chi, I + offset * 0.5) * \
                            (self.H_ext[I + offset][k] - self.H_ext[I][k])
                b[I] /= self.dx

    def build_b(self, solver : MGPCGPoissonSolver):
        self.build_b_kernel(solver.grid_type[0],
                            solver.b)
    
    @ti.kernel
    def build_A_kernel(self, 
                       level : ti.template(),
                       Adiag : ti.template(), 
                       Ax : ti.template()):
        dx = self.dx * (2 ** level)

        for I in ti.grouped(Adiag):
            for k in ti.static(range(self.dim)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(self.dim, k) * s
                        if I[k] + s >= 0 and I[k] + s < self.res[k]: # magnetic shielding
                            Adiag[I] -= (1 + utils.sample(self.chi, (I + offset * 0.5) * (2 ** level)))
                            if ti.static(s > 0): Ax[I][k] = (1 + utils.sample(self.chi, (I + offset * 0.5) * (2 ** level)))

    def build_A(self, solver : MGPCGPoissonSolver, level):
        self.build_A_kernel(level, 
                            solver.Adiag[level], 
                            solver.Ax[level])
