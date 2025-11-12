import math

import warp as wp


class Example:
    def __init__(self, verbose=False):
        self.sim_width = 128
        self.sim_height = 128

        fps = 60
        self.sim_substeps = 16
        self.sim_dt = (1.0 / fps) / self.sim_substeps
        self.sim_time = 0.0

        # wave constants
        self.k_speed = 1.0
        self.k_damp = 0.0

        # grid constants
        self.grid_size = 0.1
        self.grid_displace = 0.5

        self.verbose = verbose

        vertices = []
        self.indices = []

        def grid_index(x, y, stride):
            return y * stride + x

        for z in range(self.sim_height):
            for x in range(self.sim_width):
                pos = (
                    float(x) * self.grid_size,
                    0.0,
                    float(z) * self.grid_size,
                )

                # directly modifies verts_host memory since this is a numpy alias of the same buffer
                vertices.append(pos)

                if x > 0 and z > 0:
                    self.indices.append(grid_index(x - 1, z - 1, self.sim_width))
                    self.indices.append(grid_index(x, z, self.sim_width))
                    self.indices.append(grid_index(x, z - 1, self.sim_width))

                    self.indices.append(grid_index(x - 1, z - 1, self.sim_width))
                    self.indices.append(grid_index(x - 1, z, self.sim_width))
                    self.indices.append(grid_index(x, z, self.sim_width))

        # simulation grids
        self.sim_grid0 = wp.zeros(self.sim_width * self.sim_height, dtype=float)
        self.sim_grid1 = wp.zeros(self.sim_width * self.sim_height, dtype=float)
        self.sim_verts = wp.array(vertices, dtype=wp.vec3)

        # create surface displacement around a point
        self.cx = self.sim_width / 2 + math.sin(self.sim_time) * self.sim_width / 3
        self.cy = self.sim_height / 2 + math.cos(self.sim_time) * self.sim_height / 3

        self.renderer = None

    def step(self):
        with wp.ScopedTimer("step"):
            for _s in range(self.sim_substeps):
                # create surface displacement around a point
                self.cx = self.sim_width / 2 + math.sin(self.sim_time) * self.sim_width / 3
                self.cy = self.sim_height / 2 + math.cos(self.sim_time) * self.sim_height / 3

                # swap grids
                (self.sim_grid0, self.sim_grid1) = (self.sim_grid1, self.sim_grid0)

                self.sim_time += self.sim_dt

        with wp.ScopedTimer("mesh", self.verbose):
            # update grid vertices from heights
            pass

if __name__ == "__main__":
    example = Example()
    for _ in range(300):
        example.step()