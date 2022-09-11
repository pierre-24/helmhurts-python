import numpy

from typing import Tuple


class HelmholtzSolver:
    def __init__(
            self,
            k: float,
            n_map: numpy.ndarray,
            s_map: numpy.ndarray,
            delta: Tuple[float, float],
            absorption_layer_thickness: int = 5,
            absorption_layer_coef: float = 1.0,
    ):
        self.k = k

        self.n_map_orig = n_map
        self.s_map_orig = s_map
        self.absorption_layer_thickness = absorption_layer_thickness

        # diffraction map
        self.n_map = numpy.ones(
            (n_map.shape[0] + 2 * absorption_layer_thickness, n_map.shape[1] + 2 * absorption_layer_thickness),
            dtype=complex
        )

        self.n_map[
            absorption_layer_thickness:-absorption_layer_thickness,
            absorption_layer_thickness:-absorption_layer_thickness
        ] = self.n_map_orig  # copy original map

        self.n_map[0:absorption_layer_thickness, :] = 1 + 1j * self.k * absorption_layer_coef  # add absorption layers
        self.n_map[-absorption_layer_thickness:, :] = 1 + 1j * self.k * absorption_layer_coef
        self.n_map[:, 0:absorption_layer_thickness] = 1 + 1j * self.k * absorption_layer_coef
        self.n_map[:, -absorption_layer_thickness:] = 1 + 1j * self.k * absorption_layer_coef

        # source map
        self.s_map = numpy.zeros(
            (n_map.shape[0] + 2 * absorption_layer_thickness, n_map.shape[1] + 2 * absorption_layer_thickness))

        self.s_map[
            absorption_layer_thickness:-absorption_layer_thickness,
            absorption_layer_thickness:-absorption_layer_thickness
        ] = self.s_map_orig  # copy original map

        self.xdim, self.ydim = self.n_map.shape
        self.idelta2 = delta[0]**-2, delta[1]**-2

    def compute_E(self) -> numpy.ndarray:
        raise NotImplementedError()
