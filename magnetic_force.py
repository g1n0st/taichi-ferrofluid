import taichi as ti
from mgpcg import MGPCGPoissonSolver
from pressure_project import PressureProjectStrategy
import utils

@ti.data_oriented
class MagneticForceStrategy(PressureProjectStrategy):
    def __init__(self, dim, velocity, ghost_fluid_method, phi, p0,
                 mu0, k, H, simulator):
        super().__init__(dim, velocity, ghost_fluid_method, phi, p0)
        
        self.mu0 = mu0
        self.k = k
        self.H = H
        self.simulator = simulator

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
                            H = self.simulator.H_interp(I + offset + 0.5)
                            b[I] += self.scale_A * (self.p0 + 1/2 * self.k * self.mu0 * H.dot(H))
                            self.scale_A * (self.p0 + 1/2 * self.k * self.mu0 * H.dot(H))
