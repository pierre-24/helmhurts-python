"""
Solve the Helmholtz equation by solving a set of equations
"""

import numpy
from scipy import sparse
from scipy.sparse.linalg import spsolve as sparse_solve

from helmhurts import HelmholtzSolver

from typing import Tuple


class HelmholtzSolverSystem(HelmholtzSolver):

    def _Sf(self) -> Tuple[sparse.csc_array, numpy.ndarray]:
        length = self.mat_dim[0] * self.mat_dim[1]
        n_sparse_value = 5 * self.xdim * self.ydim + 2 * self.mat_dim[0] + 2 * self.mat_dim[1] - 2
        sparse_array_index = numpy.zeros((n_sparse_value, 2), dtype=int)
        sparse_array_value = numpy.zeros(n_sparse_value, dtype=self.n.dtype)
        i = 0

        # define the system of equation
        for x in range(1, self.xdim + 1):
            for y in range(1, self.ydim + 1):

                index_0 = numpy.ravel_multi_index((x, y), self.mat_dim)
                sparse_array_index[i] = index_0, index_0
                sparse_array_value[i] = (self.k / self.n[x - 1, y - 1]) ** 2 - 2 * self.idelta2[0] - 2 * self.idelta2[1]

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
                    index = numpy.ravel_multi_index(coo[:2], self.mat_dim)
                    sparse_array_index[i] = index_0, index
                    sparse_array_value[i] = coo[2]
                    i += 1

        # define the source
        f = numpy.insert(self.f, self.xdim, numpy.zeros(self.xdim), axis=1)
        f = numpy.insert(f, 0, numpy.zeros(self.xdim), axis=1)

        f = numpy.insert(f, self.xdim, numpy.zeros(self.ydim + 2), axis=0)
        f = numpy.insert(f, 0, numpy.zeros(self.ydim + 2), axis=0)

        f = f.ravel()

        # boundary conditions
        for j in range(0, self.xdim + 2):
            for coo in [(j, 0, self.boundary_condition[0]), (j, self.ydim + 1, self.boundary_condition[1])]:
                index = numpy.ravel_multi_index(coo[:2], self.mat_dim)
                f[index] = coo[2]
                sparse_array_index[i] = index, index
                sparse_array_value[i] = 1
                i += 1

        for j in range(1, self.ydim + 1):
            for coo in [(0, j, self.boundary_condition[2]), (self.xdim + 1, j, self.boundary_condition[3])]:
                index = numpy.ravel_multi_index(coo[:2], self.mat_dim)
                f[index] = coo[2]
                sparse_array_index[i] = index, index
                sparse_array_value[i] = 1
                i += 1

        # Create the system
        S = sparse.csc_array(
            (sparse_array_value, (sparse_array_index[:, 0], sparse_array_index[:, 1])),
            shape=(length, length)
        )

        return S, f

    def E(self) -> numpy.ndarray:
        return sparse_solve(*self._Sf()).reshape(self.mat_dim)[1:self.xdim + 1, 1: self.ydim + 1]
