import taichi as ti
import taichi_glsl as ts

import utils
from utils import *
from priority_queue import PriorityQueue

from functools import reduce

@ti.data_oriented
class LevelSet:
    def __init__(self, dim, res, dx, markers, total_mk, real):
        self.dim = dim
        self.res = res
        self.dx = dx
        self.real = real

        self.total_mk = total_mk
        self.markers = markers

        self.t = ti.field(dtype=ti.i32, shape=res) # indices to the closest points
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

    @ti.func
    def sg_to_pq(self):
        self.priority_queue.clear()
        cnt = 0
        while cnt < self.total_sg[None]:
            I = self.surface_grid[cnt]
            self.priority_queue.push(self.phi_temp[I], I)
            cnt += 1

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

        self.sg_to_pq()

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
                            if I[k1] == 0 or (I[k1] < self.res[k1] - 1 and ti.abs(self.phi_temp[I + o]) < ti.abs(self.phi_temp[I - o])): nb[k1] = ti.abs(self.phi_temp[I + o])
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

    @ti.kernel
    def distance_to_markers(self):
        # (Initialize the arrays near the input geometry)
        for p in range(self.total_mk):
            I = (self.markers[p] / self.dx).cast(int)
            d = (self.markers[p] - (I + 0.5) * self.dx).norm()
            if all(I >= 0 and I < self.res) and d < self.phi[I]:
                self.phi[I] = d
                self.t[I] = p

        # (Propagate closest point and distance estimates to the rest of the grid)
        self.total_sg[None] = 0
        for I in ti.grouped(self.phi):
            if self.t[I] != -1:
                offset = self.total_sg[None].atomic_add(1)
                self.surface_grid[offset] = I
        
        self.sg_to_pq() 
        
        while not self.priority_queue.empty():
            I0 = self.priority_queue.top()
            self.priority_queue.pop()

            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I = I0 + offset
                    if I[k] >= 0 and I[k] < self.res[k] and \
                    self.t[I] == -1:

                        for k1 in ti.static(range(self.dim)):
                            for s1 in ti.static((-1, 1)):
                                o = ti.Vector.unit(self.dim, k1) * s1
                                if I[k1] + s1 >= 0 and I[k1] + s1 < self.res[k1] and self.t[I + o] != -1:
                                    d = (self.markers[self.t[I + o]] - (I + 0.5) * self.dx).norm()
                                    if d < self.phi[I]:
                                        self.phi[I] = d
                                        self.t[I] = self.t[I + o]

                        self.priority_queue.push(self.phi[I], I)
 
    @ti.kernel
    def minus_redistance(self):
        for I in ti.grouped(self.phi):
            self.phi[I] -= (0.99 * self.dx)

        self.total_sg[None] = 0
        
        for I in ti.grouped(self.phi):
            sign_change = False
            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I1 = I + offset
                    if I1[k] >= 0 and I1[k] < self.res[k] and \
                    ts.sign(self.phi[I]) != ts.sign(self.phi[I1]):
                        sign_change = True

            if sign_change and self.phi[I] <= 0:
                self.valid[I] = 1
                self.phi_temp[I] = self.phi[I]
                offset = self.total_sg[None].atomic_add(1)
                self.surface_grid[offset] = I
            elif self.phi[I] <= 0:
                self.phi_temp[I] = ti.cast(-1, self.real)
            else:
                self.phi_temp[I] = self.phi[I]
            

        self.priority_queue.clear()
        cnt = 0
        while cnt < self.total_sg[None]:
            I = self.surface_grid[cnt]
            self.priority_queue.push(ti.abs(self.phi_temp[I]), I)
            cnt += 1

    @ti.kernel
    def smoothing(self):
        for I in ti.grouped(self.phi_temp):
            phi_avg = ti.cast(0, self.real)
            tot = ti.cast(0, int)
            for k in ti.static(range(self.dim)):
                for s in ti.static((-1, 1)):
                    offset = ti.Vector.unit(self.dim, k) * s
                    I1 = I + offset
                    if I1[k] >= 0 and I1[k] < self.res[k]:
                        phi_avg += self.phi_temp[I1]
                        tot += 1

            phi_avg /= tot
            self.phi[I] = phi_avg if phi_avg < self.phi_temp[I] else self.phi_temp[I]

    def build_from_markers(self):
        self.phi.fill(1e20)
        self.t.fill(-1)
        self.valid.fill(0)
        self.distance_to_markers()
        self.minus_redistance()
        self.propagate()
        self.smoothing()
        # self.phi.copy_from(self.phi_temp)
