import taichi as ti
from fluid_simulator import *
from initializer_3d import *
from visualizer_3d import *

if __name__ == '__main__':
    res = 128
    solver = FluidSimulator(3, (res, res, res), 0.001, 30, 0.15625, 1e3, [0, -9.8, 0])
    initializer = Initializer3D(res, 0.7, 0.5, 0.7)
    visualizer = Visualizer3D()
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
