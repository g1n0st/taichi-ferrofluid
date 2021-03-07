import taichi as ti
from fluid_simulator import *
from ferrofluid_simulator import *
from apic_extension import *
from initializer_2d import *
from visualizer_2d import *

if __name__ == '__main__':
    res = 512
    solver = FerrofluidSimulator(2, (res, res), 0.01, 1, 10 / 128)
    # initializer = SphereInitializer2D(res, 0.5, 0.5, 0.2)
    x0, y0, x1, y1 = 0.20, 0.3, 0.8, 0.5
    initializer = Initializer2D(res, x0 + 0.05, y0 + 0.05, x1 - 0.02, y1 - 0.02, \
        [[x0, y0, x1, y0 + 0.02], \
         [x0, y0 + 0.02, x0 + 0.02, y1 + 0.02],
         [x1 - 0.02, y0 + 0.02, x1, y1 + 0.02]])
    visualizer = VideoVisualizer2D(res, 512, 'visual')
    solver.initialize(initializer)
    solver.run(-1, visualizer)
    # ti.kernel_profiler_print()
