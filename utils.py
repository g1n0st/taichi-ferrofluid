import taichi as ti
import taichi_glsl as ts

FLUID = 0
AIR = 1
SOLID = 2

@ti.pyfunc
def clamp(x, a, b):
    return max(a, min(b, x))

@ti.func
def sample(data, pos):
    tot = data.shape
    # static unfold for efficiency
    if ti.static(len(data.shape) == 2):
        i, j = ts.clamp(int(pos[0]), 0, tot[0] - 1), ts.clamp(int(pos[1]), 0, tot[1] - 1)
        ip, jp = ts.clamp(i + 1, 0, tot[0] - 1), ts.clamp(j + 1, 0, tot[1] - 1)
        s, t = ts.clamp(pos[0] - i, 0.0, 1.0), ts.clamp(pos[1] - j, 0.0, 1.0)
        return \
            (data[i, j] * (1 - s) + data[ip, j] * s) * (1 - t) + \
            (data[i, jp] * (1 - s) + data[ip, jp] * s) * t

    else:
        i, j, k = ts.clamp(int(pos[0]), 0, tot[0] - 1), ts.clamp(int(pos[1]), 0, tot[1] - 1), ts.clamp(int(pos[2]), 0, tot[2] - 1)
        ip, jp, kp = ts.clamp(i + 1, 0, tot[0] - 1), ts.clamp(j + 1, 0, tot[1] - 1), ts.clamp(k + 1, 0, tot[2] - 1)
        s, t, u = ts.clamp(pos[0] - i, 0.0, 1.0), ts.clamp(pos[1] - j, 0.0, 1.0), ts.clamp(pos[2] - k, 0.0, 1.0)
        return \
            ((data[i, j, k] * (1 - s) + data[ip, j, k] * s) * (1 - t) + \
            (data[i, jp, k] * (1 - s) + data[ip, jp, k] * s) * t) * (1 - u) + \
            ((data[i, j, kp] * (1 - s) + data[ip, j, kp] * s) * (1 - t) + \
            (data[i, jp, kp] * (1 - s) + data[ip, jp, kp] * s) * t) * u

@ti.func
def splat(data, weights, v, pos, cp):
    tot = data.shape
    dim = len(tot)

    I0 = ti.Vector.zero(int, len(tot))
    I1 = ti.Vector.zero(int, len(tot))
    w = ti.zero(pos)

    for k in ti.static(range(len(tot))):
        I0[k] = ts.clamp(int(pos[k]), 0, tot[k] - 1)
        I1[k] = ts.clamp(I0[k] + 1, 0, tot[k] - 1)
        w[k] = ts.clamp(pos[k] - I0[k], 0.0, 1.0)

    for u in ti.static(ti.grouped(ti.ndrange(*((0, 2), ) * len(tot)))):
        dpos = ti.zero(pos)
        I = ti.Vector.zero(int, len(tot))
        W = 1.0
        for k in ti.static(range(len(tot))): 
            dpos[k] = (pos[k] - I0[k]) if u[k] == 0 else (pos[k] - I1[k])
            I[k] = I0[k] if u[k] == 0 else I1[k]
            W *= (1 - w[k]) if u[k] == 0 else w[k]
        data[I] += (v + cp.dot(dpos)) * W
        weights[I] += W
