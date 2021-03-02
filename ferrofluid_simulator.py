import taichi as ti
import taichi_glsl as ts

import utils
from utils import *
from mgpcg import MGPCGPoissonSolver
from pressure_project import PressureProjectStrategy
from level_set import FastMarchingLevelSet, FastSweepingLevelSet
from surface_tension import SurfaceTension
from fluid_simulator import FluidSimulator

from functools import reduce
import time
import numpy as np

@ti.data_oriented
class FerrofluidSimulator(FluidSimulator):
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

    def substep(self, dt):
        self.begin_substep(dt)

        self.solve_pressure(dt)
        if self.verbose:
            prs = np.max(self.pressure.to_numpy())
            print(f'\033[36mMax pressure: {prs}\033[0m')
        self.apply_pressure(dt)
        self.extrap_velocity()
        self.enforce_boundary()

        self.end_substep(dt) 
