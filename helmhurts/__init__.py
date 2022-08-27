import numpy

from typing import Tuple


class HelmholtzSolver:
    def __init__(
            self,
            k: float,
            n: numpy.ndarray,
            f: numpy.ndarray,
            delta: Tuple[float, float],
            boundary_cond: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        self.k = k
        self.n = n
        self.f = f
        self.boundary_condition = boundary_cond

        self.xdim, self.ydim = self.n.shape
        self.mat_dim = self.xdim + 2, self.ydim + 2
        self.idelta2 = delta[0]**-2, delta[1]**-2

    def E(self) -> numpy.ndarray:
        raise NotImplementedError()