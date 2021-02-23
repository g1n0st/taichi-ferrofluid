import taichi as ti
import taichi_glsl as ts

import utils
from utils import *
from priority_queue import PriorityQueue

from functools import reduce

@ti.data_oriented
class LevelSet:
    def __init__(self, dim, res, dx, real):
        self.dim = dim
        self.res = res
        self.dx = dx
        self.real = real

        self.phi = ti.field(dtype=real, shape=res)
        self.phi_temp = ti.field(dtype=real, shape=res)

        self.priority_queue = PriorityQueue(dim, res, real)
        self.valid = ti.field(dtype=ti.i8, shape=res)
        self.surface_grid = ti.Vector.field(dim, dtype=ti.i32, shape=reduce(lambda x, y : x * y, res))
        self.total_sg = ti.field(dtype=ti.i32, shape=())

    @ti.func
    def distance_of_aabb(self, x, x0, x1):
        phi = ti.cast(0, self.real)
        if all(x > x0) and all(x < x1): # (inside)
            phi = (ti.max(x0 - x, x - x1)).max()
        else: # (outside)
            # Find the closest point (p,q,r) on the surface of the box
            p = ti.zero(x)
            for k in ti.static(range(self.dim)):
                if x[k] < x0[k]: p[k] = x0[k]
                elif x[k] > x1[k]: p[k] = x1[k]
                else: p[k] = x[k]
            phi = (x - p).norm()

        return phi

    @ti.kernel
    def initialize_with_aabb(self, x0 : ti.template(), x1 : ti.template()):
        for I in ti.grouped(self.phi):
            self.phi[I] = self.distance_of_aabb((I + 0.5) * self.dx, x0, x1)

    @ti.kernel
    def target_surface(self):
        self.total_sg[None] = 0
        
        for I in ti.grouped(self.phi):
            sign_change = False
            est = ti.cast(1e20, self.real)
            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I1 = I + offset
                    if I1[k] >= 0 and I1[k] < self.res[k] and \
                    ts.sign(self.phi[I]) != ts.sign(self.phi[I1]):
                        theta = self.phi[I] / (self.phi[I] - self.phi[I1])
                        est0 = ts.sign(self.phi[I]) * theta * self.dx
                        est = est0 if ti.abs(est0) < ti.abs(est) else est
                        sign_change = True

            if sign_change:
                self.phi_temp[I] = est
                self.valid[I] = 1
                offset = self.total_sg[None].atomic_add(1)
                self.surface_grid[offset] = I
            else:
                self.phi_temp[I] = ti.cast(1e20, self.real) # an upper bound for all possible distances

        self.priority_queue.clear()
        cnt = 0
        while cnt < self.total_sg[None]:
            I = self.surface_grid[cnt]
            self.priority_queue.push(ti.abs(self.phi_temp[I]), I)
            cnt += 1

    # Fast Marching Method
    @ti.kernel
    def propagate(self):
        while not self.priority_queue.empty():
            I0 = self.priority_queue.top()
            self.priority_queue.pop()

            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I = I0 + offset
                    if I[k] >= 0 and I[k] < self.res[k] and \
                    self.valid[I] == 0:
                        # solve the Eikonal equation
                        nb = ti.Vector.zero(self.real, self.dim)
                        for k1 in ti.static(range(self.dim)):
                            o = ti.Vector.unit(self.dim, k1)
                            if I[k1] == 0 or ti.abs(self.phi_temp[I + o]) < ti.abs(self.phi_temp[I - o]): nb[k1] = ti.abs(self.phi_temp[I + o])
                            else: nb[k1] = ti.abs(self.phi_temp[I - o])
                   
                        # sort
                        for i in ti.static(range(self.dim-1)):
                            for j in ti.static(range(self.dim-1-i)):
                                if nb[j] > nb[j + 1]: nb[j], nb[j + 1] = nb[j + 1], nb[j]
                        
                        # (Try just the closest neighbor)
                        d = nb[0] + self.dx
                        if d > nb[1]:
                            # (Try the two closest neighbors)
                            d = (1/2) * (nb[0] + nb[1] + ti.sqrt(2 * (self.dx ** 2) - (nb[1] - nb[0]) ** 2))
                            if ti.static(self.dim == 3):
                                if d > nb[2]:
                                    # (Use all three neighbors)
                                    d = (1/3) * (nb[0] + nb[1] + nb[2] + ti.sqrt(ti.max(0, (nb[0] + nb[1] + nb[2]) ** 2 - 3 * (nb[0] ** 2 + nb[1] ** 2 + nb[2] ** 2 - self.dx ** 2))))
    
                        if d < ti.abs(self.phi_temp[I]): 
                            self.phi_temp[I] = d * ts.sign(self.phi[I0])
                        self.valid[I] = 1
                        self.priority_queue.push(ti.abs(self.phi_temp[I]), I)

    def redistance(self):
        self.valid.fill(0)
        self.target_surface()
        self.propagate()
        self.phi.copy_from(self.phi_temp)
