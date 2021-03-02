import taichi as ti
from fluid_simulator import *
from ferrofluid_simulator import *
from apic_extension import *
from initializer_2d import *
from visualizer_2d import *

if __name__ == '__main__':
    res = 128
    solver = FerrofluidSimulator(2, (res, res), 0.01, 1, 10 / res)
    initializer = Initializer2D(res, 0.1, 0.4, 0.4, 0.9, 0.44, 0.6, 0.1, 0.5)
    visualizer = Visualizer2D(res, 'levelset', False)
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
