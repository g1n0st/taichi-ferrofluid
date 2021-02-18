import taichi as ti
from mgpcg import MGPCGPoissonSolver
import utils

@ti.data_oriented
class PressureProjectStrategy:
    def __init__(self, velocity, real = float):
        self.scale_L = 1.0
        self.scale_b = 1.0
        self.velocity = velocity
        self.real = real

    @ti.kernel
    def build_b_kernel(self, res : ti.template(), dim : ti.template(), cell_type : ti.template(), b : ti.template()):
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

    def build_b(self, solver : MGPCGPoissonSolver):
        self.build_b_kernel(solver.res, solver.dim, solver.grid_type[0], solver.b)
    
    @ti.func
    def is_fluid(self, grid_type, I):
        return all(I >= 0) and all(I < grid_type.shape) and grid_type[I] == utils.FLUID

    @ti.func
    def is_solid(self, grid_type, I): # TODO: boundary condition breaks symmetry (?) 
        return any(I < 0) or any(I >= grid_type.shape) or grid_type[I] == utils.SOLID

    @ti.kernel
    def build_L_kernel(self, dim : ti.template(), l : ti.template(), grid_type : ti.template(), L : ti.template()):        
        scale = self.scale_L

        for I in ti.grouped(grid_type):
            if grid_type[I] == utils.FLUID:
                s = ti.cast(2 ** dim, self.real)
                for k in ti.static(range(dim)):
                    s -= ti.cast(self.is_solid(grid_type, I - ti.Vector.unit(dim, k)), self.real)
                    s -= ti.cast(self.is_solid(grid_type, I + ti.Vector.unit(dim, k)), self.real)
                L[I][dim * 2] = s * scale
                L[I][dim * 2 + 1] = 1 / (s * scale)
                for k in ti.static(range(dim)):
                    L[I][k * 2] = -scale * ti.cast(self.is_fluid(grid_type, I - ti.Vector.unit(dim, k)), self.real)
                    L[I][k * 2 + 1] = -scale * ti.cast(self.is_fluid(grid_type, I + ti.Vector.unit(dim, k)), self.real)

    def build_L(self, solver : MGPCGPoissonSolver, level):
        self.build_L_kernel(solver.dim, level, solver.grid_type[level], solver.L[level])
