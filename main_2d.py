import taichi as ti
from fluid_simulator import *
from ferrofluid_simulator import *
from apic_extension import *
from initializer_2d import *
from visualizer_2d import *

if __name__ == '__main__':
    res = 128
    solver = FluidSimulator(2, (res, res), 0.01, 4, 10 / res)
    # initializer = Initializer2D(res, 0.05, 0, 0.95, 0.2, 0.44, 0.6, 0.1, 0.5)
    initializer = Initializer2D(res, 0.1, 0.4, 0.4, 0.9, 0.44, 0.6, 0.1, 0.5)
    visualizer = GUIVisualizer2D(res, 512, 'visual')
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
