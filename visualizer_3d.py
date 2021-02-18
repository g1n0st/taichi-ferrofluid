import taichi as ti
import utils
from fluid_simulator import *
import numpy as np

@ti.data_oriented
class Visualizer3D: # tmp visualizer
    def __init__(self):
        self.frame = 0

    def visualize(self, simulator):
        vertices = simulator.total_mk[None]
        writer = ti.PLYWriter(num_vertices = vertices)
        pos = simulator.markers.to_numpy()
        writer.add_vertex_pos(pos[0 : vertices, 0], pos[0 : vertices, 1], pos[0 : vertices, 2])
        writer.export_frame_ascii(self.frame, "water.ply")
        self.frame += 1


