import taichi as ti

import utils
from utils import *
from mgpcg import *
from project import *

from functools import reduce
import time
import numpy as np

ti.init(arch=ti.cpu, kernel_profiler=True)

@ti.data_oriented
class FluidSimulator:
    def __init__(self,
        dim = 2,
        res = (128, 128),
        dt = 1.25e-2,
        substeps = 1,
        dx = 1.0,
        rho = 1000.0,
        gravity = [0, -9.8],
        real = float):
        
        self.dim = dim
        self.real = real
        self.res = res
        self.dx = dx
        self.dt = dt
        
        self.rho = rho
        self.gravity = gravity
        self.substeps = substeps

        # cell_type
        self.cell_type = ti.field(dtype=ti.i32)

        self.velocity = [ti.field(dtype=real) for _ in range(self.dim)] # MAC grid
        self.velocity_backup = [ti.field(dtype=real) for _ in range(self.dim)]
        self.pressure = ti.field(dtype=real)

        # extrap utils
        self.valid = ti.field(dtype=ti.i32)
        self.valid_temp = ti.field(dtype=ti.i32)

        # x/v for marker particles
        self.total_mk = ti.field(dtype=ti.i32, shape = ())
        self.markers = ti.Vector.field(dim, dtype=real)
        
        indices = ti.ijk if self.dim == 3 else ti.ij
        ti.root.dense(ti.i, reduce(lambda x, y : x * y, res) * (2 ** dim)).place(self.markers)
        ti.root.dense(indices, res).place(self.cell_type, self.pressure)
        ti.root.dense(indices, [res[_] + 1 for _ in range(self.dim)]).place(self.valid, self.valid_temp)
        for d in range(self.dim):
            ti.root.dense(indices, [res[_] + (d == _) for _ in range(self.dim)]).place(self.velocity[d], self.velocity_backup[d])

        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.iterations = 50
        self.verbose = True
        self.poisson_solver = MGPCGPoissonSolver(self.dim, 
                                                 self.res, 
                                                 self.n_mg_levels,
                                                 self.pre_and_post_smoothing,
                                                 self.bottom_smoothing,
                                                 self.real)
        
        self.strategy = PressureProjectStrategy(0, 0, self.velocity)

    @ti.func
    def is_valid(self, I):
        valid = True
        for k in ti.static(range(self.dim)):
            if I[k] < 0 or I[k] >= self.res[k]: valid = False
        return valid

    @ti.func
    def is_fluid(self, I):
        return self.is_valid(I) and self.cell_type[I] == utils.FLUID

    @ti.func
    def is_solid(self, I):
        return self.is_valid(I) and self.cell_type[I] == utils.SOLID

    @ti.func
    def is_air(self, I):
        return self.is_valid(I) and self.cell_type[I] == utils.AIR

    @ti.func
    def sample(self, data, pos, offset, tot):
        pos = pos - offset
        # static unfold for efficiency
        if ti.static(self.dim == 2):
            i, j = clamp(int(pos[0]), 0, tot[0] - 1), clamp(int(pos[1]), 0, tot[1] - 1)
            ip, jp = clamp(i + 1, 0, tot[0] - 1), clamp(j + 1, 0, tot[1] - 1)
            s, t = clamp(pos[0] - i, 0.0, 1.0), clamp(pos[1] - j, 0.0, 1.0)
            return \
                (data[i, j] * (1 - s) + data[ip, j] * s) * (1 - t) + \
                (data[i, jp] * (1 - s) + data[ip, jp] * s) * t

        else:
            i, j, k = clamp(int(pos[0]), 0, tot[0] - 1), clamp(int(pos[1]), 0, tot[1] - 1), clamp(int(pos[2]), 0, tot[2] - 1)
            ip, jp, kp = clamp(i + 1, 0, tot[0] - 1), clamp(j + 1, 0, tot[1] - 1), clamp(k + 1, 0, tot[2] - 1)
            s, t, u = clamp(pos[0] - i, 0.0, 1.0), clamp(pos[1] - j, 0.0, 1.0), clamp(pos[2] - k, 0.0, 1.0)
            return \
                ((data[i, j, k] * (1 - s) + data[ip, j, k] * s) * (1 - t) + \
                (data[i, jp, k] * (1 - s) + data[ip, jp, k] * s) * t) * (1 - u) + \
                ((data[i, j, kp] * (1 - s) + data[ip, j, kp] * s) * (1 - t) + \
                (data[i, jp, kp] * (1 - s) + data[ip, jp, kp] * s) * t) * u

    @ti.func
    def vel_interp(self, pos):
        v = ti.Vector.zero(self.real, self.dim)
        for k in ti.static(range(self.dim)):
            v[k] = self.sample(self.velocity[k], pos / self.dx, 0.5 * (1 - ti.Vector.unit(self.dim, k)), self.velocity[k].shape)
        return v

    @ti.kernel
    def advect_markers(self, dt : ti.f32):
        for p in range(self.total_mk[None]):
            midpos = self.markers[p] + self.vel_interp(self.markers[p]) * (0.5 * dt)
            self.markers[p] += self.vel_interp(midpos) * dt

    @ti.kernel
    def apply_markers(self):
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] != utils.SOLID:
                self.cell_type[I] = utils.AIR

        for p in range(self.total_mk[None]):
            I = ti.Vector.zero(ti.i32, self.dim)
            for k in ti.static(range(self.dim)):
                I[k] = clamp(int(self.markers[p][k] / self.dx), 0, self.res[k] - 1)
            if self.cell_type[I] != utils.SOLID:
                self.cell_type[I] = utils.FLUID

    @ti.kernel
    def add_gravity(self, dt : ti.f32):
        for k in ti.static(range(self.dim)):
            if ti.static(self.gravity[k] != 0):
                g = self.gravity[k]
                for I in ti.grouped(self.velocity[k]):
                    self.velocity[k][I] += g * dt
    
    @ti.kernel
    def enforce_boundary(self):
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == utils.SOLID:
                for k in ti.static(range(self.dim)):
                    self.velocity[k][I] = 0
                    self.velocity[k][I + ti.Vector.unit(self.dim, k)] = 0

    def solve_pressure(self, dt):
        self.strategy.scale_A = dt / (self.rho * self.dx * self.dx)
        self.strategy.scale_b = 1 / self.dx

        start1 = time.perf_counter()
        self.poisson_solver.reinitialize(self.cell_type, self.strategy)
        end1 = time.perf_counter()

        start2 = time.perf_counter()
        self.poisson_solver.solve(self.iterations, self.verbose)
        end2 = time.perf_counter()

        print(f'\033[33minit cost {end1 - start1}s, solve cost {end2 - start2}s\033[0m')
        self.pressure.copy_from(self.poisson_solver.x)

    @ti.kernel
    def apply_pressure(self, dt : ti.f32):
        scale = dt / (self.rho * self.dx)

        for k in ti.static(range(self.dim)):
            for I in ti.grouped(self.cell_type):
                I_1 = I - ti.Vector.unit(self.dim, k)
                if self.is_fluid(I_1) or self.is_fluid(I):
                    if self.is_solid(I_1) or self.is_solid(I): self.velocity[k][I] = 0
                    else: self.velocity[k][I] += scale * (self.pressure[I_1] - self.pressure[I])
    @ti.func
    def advect(self, I, dst, src, offset, dt):
        pos = (I + offset) * self.dx
        midpos = pos - self.vel_interp(pos) * (0.5 * dt)
        p0 = pos - self.vel_interp(midpos) * dt
        dst[I] = self.sample(src, p0 / self.dx, offset, src.shape)

    @ti.kernel
    def advect_velocity(self, dt : ti.f32):
        for k in ti.static(range(self.dim)):
            offset = 0.5 * (1 - ti.Vector.unit(self.dim, k))
            for I in ti.grouped(self.velocity_backup[k]):
                self.advect(I, self.velocity_backup[k], self.velocity[k], offset, dt)

    def update_velocity(self):
        for k in range(self.dim):
            self.velocity[k].copy_from(self.velocity_backup[k]) 

    @ti.kernel
    def mark_valid(self, k : ti.template()):
        for I in ti.grouped(self.velocity[k]):
            # NOTE that the the air-liquid interface is valid
            I_1 = I - ti.Vector.unit(self.dim, k)
            if self.is_fluid(I_1) or self.is_fluid(I):
                self.valid[I] = 1
            else:
                self.valid[I] = 0

    @ti.kernel
    def diffuse_quantity(self, dst : ti.template(), src : ti.template(), valid_dst : ti.template(), valid : ti.template()):
        for I in ti.grouped(dst):
            if valid[I] == 0:
                tot = ti.cast(0, self.real)
                cnt = 0
                for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 2), ) * self.dim))):
                    if valid[I + offset] == 1:
                        tot += src[I + offset]
                        cnt += 1
                if cnt > 0:
                    dst[I] = tot / ti.cast(cnt, self.real)
                    valid_dst[I] = 1

    def extrap_velocity(self):
        for k in range(self.dim):
            self.mark_valid(k)
            for i in range(10):
                self.velocity_backup[k].copy_from(self.velocity[k])
                self.valid_temp.copy_from(self.valid)
                self.diffuse_quantity(self.velocity[k], self.velocity_backup[k], self.valid, self.valid_temp)

    def substep(self, dt):
        self.advect_markers(dt)
        self.apply_markers()

        self.advect_velocity(dt)
        self.update_velocity()
        self.enforce_boundary()

        mks = max(np.max(self.velocity[0].to_numpy()), np.max(self.velocity[1].to_numpy()))
        print(f'\033[36mMax advect velocity: {mks}\033[0m')

        self.add_gravity(dt)
        self.enforce_boundary()

        self.extrap_velocity()
        self.enforce_boundary()

        self.solve_pressure(dt)
        
        prs = np.max(self.pressure.to_numpy())
        print(f'\033[36mMax pressure: {prs}\033[0m')

        self.apply_pressure(dt)

        self.extrap_velocity()
        self.enforce_boundary()
    
    def run(self, max_steps, visualizer):
        step = 0
        while step < max_steps or max_steps == -1:
            print(step)
            for substep in range(self.substeps):
                self.substep(self.dt)
            visualizer.visualize(self)
            step += 1

    @ti.kernel
    def init_boundary(self):
        for I in ti.grouped(self.cell_type):
            if any(I == 0) or any(I + 1 == self.res):
                self.cell_type[I] = utils.SOLID

    @ti.kernel
    def init_markers(self):
        self.total_mk[None] = 0
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == utils.FLUID:
                for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                    num = ti.atomic_add(self.total_mk[None], 1)
                    self.markers[num] = (I + (offset + [ti.random() for _ in ti.static(range(self.dim))]) / 2) * self.dx

    @ti.kernel
    def reinitialize(self):
        for I in ti.grouped(ti.ndrange(* [self.res[_] for _ in range(self.dim)])):
            self.cell_type[I] = 0
            self.pressure[I] = 0
            for k in ti.static(range(self.dim)):
                I_1 = I + ti.Vector.unit(self.dim, k)
                self.velocity[k][I] = 0
                self.velocity[k][I_1] = 0
                self.velocity_backup[k][I] = 0
                self.velocity_backup[k][I_1] = 0

    def initialize(self, initializer):
        self.reinitialize()

        self.cell_type.fill(utils.AIR)
        initializer.init_scene(self) 
        
        self.init_boundary()
        self.init_markers()

@ti.data_oriented
class Initializer: # tmp initializer
    def __init__(self, x, y, x1, x2, y1, y2):
        self.x = x
        self.y = y
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    @ti.kernel
    def init_kernel(self, cell_type : ti.template(), dx : ti.template()):
        xn = int(self.x)
        yn = int(self.y)
        x1_ = int(self.x1)
        x2_ = int(self.x2)
        y1_ = int(self.y1)
        y2_ = int(self.y2)

        for i, j in cell_type:
            if i <= xn and j <= yn:
                cell_type[i, j] = utils.FLUID
            elif i >= x1_ and i <= x2_ and j >= y1_ and j <= y2_:
                cell_type[i, j] = utils.SOLID

    def init_scene(self, simulator):
        self.init_kernel(simulator.cell_type, simulator.dx)

@ti.data_oriented
class Visualizer: # tmp visualizer
    def __init__(self, res, switch = 1):
        self.grid_res = res
        self.res = 512
        self.switch = switch
        self.gui = ti.GUI("demo", (self.res, self.res))
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
    
    @ti.kernel
    def fill_marker(self, dx : ti.template(), p : ti.template()):
        for i, j in self.color_buffer:
            x = int((i + 0.5) / self.res * dx)
            y = int((j + 0.5) / self.res * dx)

            m = (ti.log(min(p[x, y], 100) + 1) / ti.log(10)) / 2.001
            self.color_buffer[i, j] = ti.Vector([m, m, m])

    def visualize_1(self, simulator):
        self.fill_marker(simulator.dx * 128, simulator.pressure)
        img = self.color_buffer.to_numpy()
        self.gui.set_image(img)
        self.gui.show()
    
    def visualize_2(self, simulator):
        bg_color = 0x000000
        particle_color = 0x0FFFFF
        particle_radius = 1.0
        pos = simulator.markers.to_numpy()
        
        self.gui.clear(bg_color)
        self.gui.circles(pos / (self.grid_res * simulator.dx), radius=particle_radius, color=particle_color)
        self.gui.show()

    def visualize(self, simulator):
        if self.switch == 1:
            self.visualize_1(simulator)
        else:
            self.visualize_2(simulator)

if __name__ == '__main__':
    res = 256
    solver = FluidSimulator(2, (res, res), 0.01, 1, 10 / res)
    initializer = Initializer(0.4 * res, 0.4 * res, 0.44 * res, 0.6 * res, 0.1 * res, 0.5 * res)
    visualizer = Visualizer(res, 2)
    solver.initialize(initializer)
    solver.run(-1, visualizer)
