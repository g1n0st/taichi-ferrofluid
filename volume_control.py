import taichi as ti
from mgpcg import MGPCGPoissonSolver
from pressure_project import PressureProjectStrategy
import utils

@ti.data_oriented
class PressureProjectWithVolumeControlStrategy(PressureProjectStrategy):
    def __init__(self, dim, velocity, ghost_fluid_method, phi, p0, level_set, dt):
        super().__init__(dim, velocity, ghost_fluid_method, phi, p0)
        self.level_set = level_set
        self.dt = dt
        
        self.step = 0
        self.vol_0 = 0.0 # the desired value
        # proportional-integral (PI) controller
        self.y = 0.0
        self.n_p = 25
        self.k_p = 2.3 / (self.n_p * self.dt) # proportional gain
        self.zeta = 2 # suppress the noise coming from the volume computation
        self.k_i = (self.k_p / (2 * self.zeta)) ** 2 # PI gain

    @ti.kernel
    def calc_volume(self) -> ti.f32:
        vol = ti.cast(0, ti.f32)
        for I in ti.grouped(self.phi):
            cell_vol = ti.cast(0, ti.f32)
            for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                cell_vol += self.level_set.theta(-utils.sample(self.phi, I + offset - 0.5))
            vol += cell_vol / (2 ** self.dim)

        return vol

    @ti.kernel
    def add_c(self, cell_type : ti.template(), b : ti.template(), c : ti.f32):
        for I in ti.grouped(b):
            if cell_type[I] == utils.FLUID:
                b[I] += c

    def build_b(self, solver : MGPCGPoissonSolver):
        self.step += 1
        super().build_b(solver)
        vol = self.calc_volume()
        if self.step == 1:
            self.vol_0 = vol
            return

        x = (vol - self.vol_0) / self.vol_0 # the normalized difference between the current and desired volume
        self.y += x * self.dt # drift error can be removed by integrating the volume error over time
        c = (-self.k_p * x - self.k_i * self.y) / (x + 1) # the required divergence
        print(f'\033[31mvolume = {vol}, c = {c}\033[0m')
        self.add_c(solver.grid_type[0], solver.b, c)
