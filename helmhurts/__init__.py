import numpy

from typing import Tuple


class HelmholtzSolver:
    def __init__(
            self,
            k: float,
            n_map: numpy.ndarray,
            f_map: numpy.ndarray,
            delta: Tuple[float, float]):
        self.k = k
        self.n_map = n_map
        self.f_map = f_map

        self.xdim, self.ydim = self.n_map.shape
        self.idelta2 = delta[0]**-2, delta[1]**-2

    def compute_E(self) -> numpy.ndarray:
        raise NotImplementedError()


class HelmholtzSolverDirichlet(HelmholtzSolver):
    def __init__(
            self,
            k: float,
            n_map: numpy.ndarray,
            f_map: numpy.ndarray,
            delta: Tuple[float, float],
            boundary_conditions: Tuple[float, float, float, float] = (.0, .0, .0, .0)
    ):
        super().__init__(k, n_map, f_map, delta)
        self.boundary_conditions = boundary_conditions


class HelmholtzSolverNeumann(HelmholtzSolver):
    def __init__(
            self,
            k: float,
            n_map: numpy.ndarray,
            f_map: numpy.ndarray,
            delta: Tuple[float, float],
            reflective_boundaries: bool = True
    ):
        super().__init__(k, n_map, f_map, delta)
        self.reflective_boundaries = reflective_boundaries
