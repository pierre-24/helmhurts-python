"""
Solve the Helmholtz equation by solving a set of equations
"""

import numpy
from scipy import sparse
from scipy.sparse.linalg import spsolve as sparse_solve

from helmhurts import HelmholtzSolver

from typing import Tuple


class HelmholtzSolverDirichletSystem(HelmholtzSolver):
    def __init__(
            self,
            k: float,
            n_map: numpy.ndarray,
            s_map: numpy.ndarray,
            delta: Tuple[float, float],
            absorption_layer_thickness: int = 5,
            absorption_layer_coef: float = 10,
            boundary_conditions: Tuple[float, float, float, float] = (.0, .0, .0, .0)
    ):
        super().__init__(k, n_map, s_map, delta, absorption_layer_thickness, absorption_layer_coef)
        self.boundary_conditions = boundary_conditions

    def _As(self) -> Tuple[sparse.csc_array, numpy.ndarray]:
        matrix_dim = self.xdim + 2, self.ydim + 2
        length = matrix_dim[0] * matrix_dim[1]

        n_sparse_value = 5 * self.xdim * self.ydim + 2 * matrix_dim[0] + 2 * matrix_dim[1] - 2
        sparse_array_index = numpy.zeros((n_sparse_value, 2), dtype=int)
        sparse_array_value = numpy.zeros(n_sparse_value, dtype=self.n_map.dtype)
        i = 0

        # define the system of equation
        for x in range(1, self.xdim + 1):
            for y in range(1, self.ydim + 1):

                index_0 = numpy.ravel_multi_index((x, y), matrix_dim)
                sparse_array_index[i] = index_0, index_0
                sparse_array_value[i] = \
                    (self.k * self.n_map[x - 1, y - 1].real) ** 2 - 1j * self.k * self.n_map[x - 1, y - 1].imag \
                    - 2 * self.idelta2[0] - 2 * self.idelta2[1]

                xm = x - 1
                xp = x + 1
                ym = y - 1
                yp = y + 1
                i += 1

                for coo in [
                    (xm, y, self.idelta2[0]),
                    (xp, y, self.idelta2[0]),
                    (x, ym, self.idelta2[1]),
                    (x, yp, self.idelta2[1])
                ]:
                    index = numpy.ravel_multi_index(coo[:2], matrix_dim)
                    sparse_array_index[i] = index_0, index
                    sparse_array_value[i] = coo[2]
                    i += 1

        # define the source
        f = numpy.insert(self.s_map, self.xdim, numpy.zeros(self.xdim), axis=1)
        f = numpy.insert(f, 0, numpy.zeros(self.xdim), axis=1)

        f = numpy.insert(f, self.xdim, numpy.zeros(self.ydim + 2), axis=0)
        f = numpy.insert(f, 0, numpy.zeros(self.ydim + 2), axis=0)

        f = -f.ravel()

        # boundary conditions
        for j in range(0, self.xdim + 2):
            for coo in [(j, 0, self.boundary_conditions[0]), (j, self.ydim + 1, self.boundary_conditions[1])]:
                index = numpy.ravel_multi_index(coo[:2], matrix_dim)
                f[index] = coo[2]
                sparse_array_index[i] = index, index
                sparse_array_value[i] = 1
                i += 1

        for j in range(1, self.ydim + 1):
            for coo in [(0, j, self.boundary_conditions[2]), (self.xdim + 1, j, self.boundary_conditions[3])]:
                index = numpy.ravel_multi_index(coo[:2], matrix_dim)
                f[index] = coo[2]
                sparse_array_index[i] = index, index
                sparse_array_value[i] = 1
                i += 1

        # Create the system
        A = sparse.csc_array(
            (sparse_array_value, (sparse_array_index[:, 0], sparse_array_index[:, 1])),
            shape=(length, length)
        )

        return A, f

    def compute_E(self) -> numpy.ndarray:
        return sparse_solve(*self._As()).reshape((self.xdim + 2, self.ydim + 2))[
               1+self.absorption_layer_thickness:-1-self.absorption_layer_thickness,
               1+self.absorption_layer_thickness:-1-self.absorption_layer_thickness
        ]
