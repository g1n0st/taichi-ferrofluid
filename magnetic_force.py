import taichi as ti
from mgpcg import MGPCGPoissonSolver
from pressure_project import PressureProjectStrategy
import utils

@ti.data_oriented
class MagneticForceStrategy(PressureProjectStrategy):
    def __init__(self, dim, velocity, ghost_fluid_method, phi, p0,
                 mu0, k, H):
        super().__init__(dim, velocity, ghost_fluid_method, phi, p0)
        
        self.mu0 = mu0
        self.k = k
        self.H = H

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
                                b[I] += self.scale_A * (self.p0 + 1/2 * self.k * self.mu0 * 
                                    (utils.sample(self.H, I + offset * 0.5).norm() ** 2))
                            else:
                                c = (self.phi[I] - self.phi[I + offset]) / self.phi[I]
                                c = utils.clamp(c, -1e3, 1e3)
                                b[I] += self.scale_A * c * (self.p0 + 1/2 * self.k * self.mu0 * 
                                    (utils.sample(self.H, I + offset / c).norm() ** 2))


