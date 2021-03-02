import taichi as ti
import taichi_glsl as ts

from fluid_simulator import FluidSimulator
from utils import *
import utils

from functools import reduce
import time
import numpy as np

# APIC: use apic [Jiang et al. 2015] to advect grids, advect markers with Semi-Lagrangian, then build the level-set from markers
@ti.data_oriented
class APICSimulator(FluidSimulator):
    def __init__(self,
        dim = 2,
        res = (128, 128),
        dt = 1.25e-2,
        substeps = 1,
        dx = 1.0,
        rho = 1000.0,
        gravity = [0, -9.8],
        p0 = 1e-3,
        real = float):
            super().__init__(dim, res, dt, substeps, dx, rho, gravity, p0, real)

            self.p_v = ti.Vector.field(dim, dtype=real) # velocities
            self.p_cp = [ti.Vector.field(dim, dtype=real) for _ in range(self.dim)] # affine-velocities

            max_particles = reduce(lambda x, y : x * y, res) * (2 ** dim)
            ti.root.dense(ti.i, max_particles).place(self.p_v)
            for d in range(self.dim):
                ti.root.dense(ti.i, max_particles).place(self.p_cp[d])

    @ti.func
    def apic_c(self, stagger, xp, grid_v):
        base = (xp / self.dx - stagger).cast(int)
        new_c = ti.Vector.zero(self.real, self.dim)

        for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
            dpos = xp / self.dx - (base + offset + stagger)
            weight = ti.cast(1.0, self.real)
            for k in ti.static(range(self.dim)): weight *= (1.0 - ti.abs(dpos[k]))
            new_c += 4 * weight * dpos * grid_v[base + offset]

        return new_c

    @ti.kernel
    def update_from_grid(self):
        for p in range(self.total_mk[None]):
            self.p_v[p] = self.vel_interp(self.p_x[p])
            for k in ti.static(range(self.dim)):
                self.p_cp[k][p] = self.apic_c(0.5 * (1 - ti.Vector.unit(self.dim, k)), self.p_x[p], self.velocity[k])

    @ti.kernel
    def transfer_to_grid(self):
        for p in range(self.total_mk[None]):
            for k in ti.static(range(self.dim)):
                utils.splat(self.velocity[k], self.velocity_backup[k], self.p_v[p][k], self.p_x[p] / self.dx - 0.5 * (1 - ti.Vector.unit(self.dim, k)), self.p_cp[k][p])

        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.velocity_backup[k]): # reuse velocity_backup as weight
                if self.velocity_backup[k][I] > 0:
                    self.velocity[k][I] /= self.velocity_backup[k][I]

    def substep(self, dt):
        self.level_set.build_from_markers()
        self.apply_markers()

        for k in range(self.dim):
            self.velocity[k].fill(0)
            self.velocity_backup[k].fill(0)
        self.transfer_to_grid()
        self.extrap_velocity()
        self.enforce_boundary()

        self.add_gravity(dt)
        self.enforce_boundary()

        self.solve_pressure(dt)

        if self.verbose:
            prs = np.max(self.pressure.to_numpy())
            print(f'\033[36mMax pressure: {prs}\033[0m')

        self.apply_pressure(dt)
        self.update_from_grid()
        self.advect_markers(dt)

        self.total_t += self.dt

    @ti.kernel
    def init_markers(self):
        self.total_mk[None] = 0
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == utils.FLUID:
                for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                    num = ti.atomic_add(self.total_mk[None], 1)
                    self.p_x[num] = (I + (offset + [ti.random() for _ in ti.static(range(self.dim))]) / 2) * self.dx
                    self.p_v[num] = ti.zero(self.p_v[num])
                    for k in ti.static(range(self.dim)):
                        self.p_cp[k][num] = ti.zero(self.p_cp[k][num])

