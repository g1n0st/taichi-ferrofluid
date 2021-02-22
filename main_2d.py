import taichi as ti
from fluid_simulator import *
from initializer_2d import *
from visualizer_2d import *

if __name__ == '__main__':
    res = 256
    solver = FluidSimulator(2, (res, res), 0.01, 1, 10 / res)
    initializer = Initializer2D(res, 0.4, 0.4, 0.44, 0.6, 0.1, 0.5)
    visualizer = Visualizer2D(res, 'levelset')
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
