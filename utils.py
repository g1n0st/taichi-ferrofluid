import taichi as ti

FLUID = 0
AIR = 1
SOLID = 2

@ti.pyfunc
def clamp(x, a, b):
    return max(a, min(b, x))
