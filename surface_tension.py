import taichi as ti
import taichi_glsl as ts

from mgpcg import MGPCGPoissonSolver

import time
import utils
from utils import *

@ti.data_oriented
class SurfaceTensionStrategy:
    def __init__(self, simulator, surface_tension):
        self.dim = simulator.dim
        self.res = simulator.res
        self.dt = simulator.dt
        self.dx = simulator.dx
        self.real = simulator.real
        self.sigma = surface_tension.sigma

        self.simulator = simulator
        self.surface_tension = surface_tension

    @ti.kernel
    def build_b_kernel(self, velocity: ti.template(), kappa : ti.template(), n : ti.template(), b : ti.template()):
        offset = 0.5 * (1 - ti.Vector.unit(self.dim, self.d))
        for I in ti.grouped(b):
            b[I] = velocity[I] / self.dt
            scale = self.sigma * self.simulator.level_set.delta( \
                        utils.sample(self.simulator.level_set.phi, I + offset))

            # calculate Du/Dn
            grad_u = ti.Vector.zero(self.real, self.dim)
            for k in ti.static(range(self.dim)):
                unit = ti.Vector.unit(self.dim, k)
                # grad_u[k] -= velocity[I]
                # if I[k] + 1 < self.res[k]: grad_u[k] += velocity[I + unit]
                # grad_u[k] /= self.dx
                grad_u[k] += (utils.sample(velocity, I + unit * 0.5) - \
                                utils.sample(velocity, I - unit * 0.5)) / self.dx
            Du_Dn = grad_u.dot(n[I])
            
            # calculate D2u/Dn2
            D2 = ti.Matrix.zero(self.real, self.dim, self.dim)
            for k in ti.static(range(self.dim)):
                unit = ti.Vector.unit(self.dim, k)
                if I[k] - 1 >= 0: D2[k, k] += velocity[I - unit] - velocity[I]
                if I[k] + 1 < self.res[k]: D2[k, k] += velocity[I + unit] - velocity[I]
                D2[k, k] /= (self.dx ** 2)
            
            for k1 in ti.static(range(self.dim)):
                for k2 in ti.static(range(self.dim)):
                    unit1 = ti.Vector.unit(self.dim, k1)
                    unit2 = ti.Vector.unit(self.dim, k2)
                    # v00 = velocity[I]
                    # v10 = velocity[I + unit1] if I[k1] + 1 < self.res[k1] else 0
                    # v01 = velocity[I + unit2] if I[k2] + 1 < self.res[k2] else 0
                    # v11 = velocity[I + unit1 + unit2] if I[k1] + 1 < self.res[k1] and I[k2] + 1 < self.res[k2] else 0
                    v00 = utils.sample(velocity, I - unit1 * 0.5 - unit2 * 0.5)
                    v10 = utils.sample(velocity, I + unit1 * 0.5 - unit2 * 0.5)
                    v01 = utils.sample(velocity, I - unit1 * 0.5 + unit2 * 0.5)
                    v11 = utils.sample(velocity, I + unit1 * 0.5 + unit2 * 0.5)

                    D2[k1, k2] = (v11 + v00 - v10 - v01) / (self.dx ** 2)
            D2u_Dn2 = (n[I].transpose() @ D2 @ n[I])[0, 0]

            b[I] -= scale * (kappa[I] * n[I][self.d] + self.dt * (D2u_Dn2 + kappa[I] * Du_Dn))

    def build_b(self, solver : MGPCGPoissonSolver):
        self.build_b_kernel(self.simulator.velocity[self.d], self.surface_tension.kappa, self.surface_tension.n, solver.b)
    
    @ti.kernel
    def build_A_kernel(self, Adiag : ti.template(), Ax : ti.template(), level : ti.template()):
        dx = self.dx * (2 ** level)
        for I in ti.grouped(Adiag):
            Adiag[I] = 1 / self.dt

        offset = 0.5 * (1 - ti.Vector.unit(self.dim, self.d))
        scale = self.sigma * self.dt / (dx ** 2)
        for I in ti.grouped(Adiag):
            for k in ti.static(range(self.dim)):
                unit = ti.Vector.unit(self.dim, k)
                if I[k] + 1 < self.res[k]:
                    Ax[I][k] = -scale * self.simulator.level_set.delta( \
                        utils.sample(self.simulator.level_set.phi, (I + unit + offset) * (2 ** level)))
                    Adiag[I] += scale * self.simulator.level_set.delta( \
                        utils.sample(self.simulator.level_set.phi, (I + offset) * (2 ** level)))
                if I[k] - 1 >= 0:
                    Adiag[I] += scale * self.simulator.level_set.delta( \
                        utils.sample(self.simulator.level_set.phi, (I + offset) * (2 ** level)))

    def build_A(self, solver : MGPCGPoissonSolver, level):
        self.build_A_kernel(solver.Adiag[level], solver.Ax[level], level)

@ti.data_oriented
class SurfaceTension:
    def __init__(self, simulator):
        self.dim = simulator.dim
        self.res = simulator.res
        self.dx = simulator.dx
        self.real = simulator.real

        self.sigma = 7.28e-2 # surface tension coefficient

        self.simulator = simulator
        self.level_set = simulator.level_set

        self.poisson_solve_iterations = 500
        self.poisson_solver = simulator.poisson_solver

        self.n = ti.Vector.field(self.dim, dtype=self.real, shape=self.res) # n = grad phi / |grad phi|
        self.kappa = ti.field(dtype=self.real, shape=self.res) # mean curvature, kappa = div n

        self.strategy = SurfaceTensionStrategy(simulator, self)

    @ti.kernel
    def calc_n(self, d : ti.template(), phi : ti.template()):
        offset = 0.5 * (1 - ti.Vector.unit(self.dim, d))
        for I in ti.grouped(self.n):
            # self.kappa[I] = ti.zero(self.kappa[I])
            for k in ti.static(range(self.dim)):
                unit = ti.Vector.unit(self.dim, k)
                self.n[I][k] = \
                                (utils.sample(phi, I + unit * 0.5 + offset) - \
                                utils.sample(phi, I - unit * 0.5 + offset)) / self.dx

            norm = self.n[I].norm()
            if norm > 0: 
                self.n[I] /= norm # conditional normalize for numerical safety

    @ti.kernel
    def calc_kappa(self, d : ti.template(), phi : ti.template()):
        for I in ti.grouped(self.kappa):
            self.kappa[I] = ti.zero(self.kappa[I])
            for k in ti.static(range(self.dim)):
                unit = ti.Vector.unit(self.dim, k)
                self.kappa[I] += (utils.sample(self.n, I + unit * 0.5) - \
                                  utils.sample(self.n, I - unit * 0.5))[k] / self.dx

    def solve_surface_tension(self):
        for k in range(self.dim):
            self.calc_n(k, self.level_set.phi)
            self.calc_kappa(k, self.level_set.phi)
            self.strategy.d = k

            start1 = time.perf_counter()
            self.poisson_solver.full_reinitialize(self.strategy)
            end1 = time.perf_counter()

            start2 = time.perf_counter()
            self.poisson_solver.solve(self.poisson_solve_iterations, self.simulator.verbose)
            end2 = time.perf_counter()

            print(f'\033[33msolve surface tension ({k}), init cost {end1 - start1}s, solve cost {end2 - start2}s\033[0m')
            utils.copy(self.poisson_solver.x, self.simulator.velocity[k])
