import taichi as ti
import utils

@ti.data_oriented
class MGPCGPoissonSolver:
    def __init__(self, dim, res, n_mg_levels = 4, pre_and_post_smoothing = 2, bottom_smoothing = 50, real = float):

        self.FLUID = utils.FLUID
        self.SOLID = utils.SOLID
        self.AIR = utils.AIR

        # grid parameters
        self.dim = dim
        self.res = res
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing
        self.real = real

        # rhs of linear system
        self.b = ti.field(dtype=real, shape=res) # Ax=b

        self.r = [ti.field(dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # residual
        self.z = [ti.field(dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # M^-1 self.r

        # grid type
        self.grid_type = [ti.field(dtype=ti.i8, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)]

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [ti.field(dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # A(i,j,k)(i,j,k)
        self.Ax = [ti.Vector.field(dim, dtype=real, shape=[res[_] // 2**l for _ in range(dim)]) 
                        for l in range(self.n_mg_levels)] # Ax=A(i,j,k)(i+1,j,k), Ay=A(i,j,k)(i,j+1,k), Az=A(i,j,k)(i,j,k+1)
        
        
        self.x = ti.field(dtype=real, shape=res) # solution
        self.p = ti.field(dtype=real, shape=res) # conjugate gradient
        self.Ap = ti.field(dtype=real, shape=res) # matrix-vector product
        self.sum = ti.field(dtype=real, shape=()) # storage for reductions
        self.alpha = ti.field(dtype=real, shape=()) # step size
        self.beta = ti.field(dtype=real, shape=()) # step size

    @ti.kernel
    def init_gridtype(self, grid0 : ti.template(), grid : ti.template()):
        for I in ti.grouped(grid):
            I2 = I * 2

            tot_fluid = 0
            tot_air = 0
            for offset in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * self.dim))):
                attr = int(grid0[I2 + offset])
                if attr == self.AIR: tot_air += 1
                elif attr == self.FLUID: tot_fluid += 1

            if tot_air > 0: grid[I] = self.AIR
            elif tot_fluid > 0: grid[I] = self.FLUID
            else: grid[I] = self.SOLID

    
    @ti.kernel
    def initialize(self):
        for I in ti.grouped(ti.ndrange(* [self.res[_] for _ in range(self.dim)])):
            self.r[0][I] = 0
            self.z[0][I] = 0
            self.Ap[I] = 0
            self.p[I] = 0
            self.x[I] = 0
            self.b[I] = 0

        for l in ti.static(range(self.n_mg_levels)):
            for I in ti.grouped(ti.ndrange(* [self.res[_] // (2**l) for _ in range(self.dim)])):
                self.grid_type[l][I] = 0
                self.Adiag[l][I] = 0
                self.Ax[l][I] = ti.zero(self.Ax[l][I])

    def reinitialize(self, cell_type, strategy):
        self.initialize()
        self.grid_type[0].copy_from(cell_type)
        strategy.build_b(self)
        strategy.build_A(self, 0)

        for l in range(1, self.n_mg_levels):
            self.init_gridtype(self.grid_type[l - 1], self.grid_type[l])
            strategy.build_A(self, l)

    @ti.func
    def neighbor_sum(self, Ax, x, I):
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += Ax[I - offset][i] * x[I - offset] + Ax[I][i] * x[I + offset]
        return ret

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase and self.grid_type[l][I] == self.FLUID:
                self.z[l][I] = (self.r[l][I] - self.neighbor_sum(self.Ax[l], self.z[l], I)) / self.Adiag[l][I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            if self.grid_type[l][I] == self.FLUID:
                Az = self.Adiag[l][I] * self.z[l][I]
                Az += self.neighbor_sum(self.Ax[l], self.z[l], I)
                res = self.r[l][I] - Az
                self.r[l + 1][I // 2] += res

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,
              max_iters=-1,
              verbose=False,
              rel_tol=1e-12,
              abs_tol=1e-14,
              eps=1e-12):

        self.r[0].copy_from(self.b)
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        if verbose:
             print(f"init rtr = {initial_rTr}")

        tol = max(abs_tol, initial_rTr * rel_tol)

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        self.v_cycle()
        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # Conjugate gradients
        iter = 0
        while max_iters == -1 or iter < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            # self.r = self.r - self.alpha self.Ap
            self.update_xr()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]

            if verbose:
                print(f'iter {iter}, |residual|_2={ti.sqrt(rTr)}')

            if rTr < tol:
                break

            # self.z = M^-1 self.r
            self.v_cycle()

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            iter += 1

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            if self.grid_type[0][I] == self.FLUID:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            if self.grid_type[0][I] == self.FLUID:
                r = self.Adiag[0][I] * self.p[I]
                r += self.neighbor_sum(self.Ax[0], self.p, I)
                self.Ap[I] = r

    @ti.kernel
    def update_xr(self):
        alpha = self.alpha[None]
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == self.FLUID:
                self.x[I] += alpha * self.p[I]
                self.r[0][I] -= alpha * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if self.grid_type[0][I] == self.FLUID:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]
