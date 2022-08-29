import numpy

from helmhurts.solver_system import HelmholtzSolverNeumannSystem

import matplotlib.pyplot as plt

sz = (512, 512)
layer_size = 10

# diffraction
n = numpy.ones(sz, dtype=complex)


n[0:layer_size, :] = n[sz[0]-layer_size-1:sz[0]-1, :] \
    = n[:, 0: layer_size] = n[:, sz[1]-layer_size-1:sz[1]-1] = 2.24 - 0.021j

n[100:120, 75:] = 2.24 - 0.021j  # wall

# source
f = numpy.zeros(sz)
f[15:25, 15:25] = 1

solver = HelmholtzSolverNeumannSystem(52, n, f, (.01, .01), reflective_boundaries=False)
E = solver.compute_E()

Er = E.real

# plot it
fig = plt.figure(figsize=(10, 8))
axes = fig.add_subplot()
axes.axis([-1, 1, -1, 1])

ve = numpy.max([-numpy.min(Er), numpy.max(Er)])

X, Y = numpy.meshgrid(numpy.linspace(-1, 1, sz[0]), numpy.linspace(-1, 1, sz[1]))

c = axes.pcolormesh(X.T, Y.T, Er, vmin=-ve, vmax=ve)
fig.colorbar(c)

fig.show()

# plot signal power
Ep = 20 * numpy.log10(E.real ** 2)
fig = plt.figure(figsize=(10, 8))
axes = fig.add_subplot()
axes.axis([-1, 1, -1, 1])

c = axes.pcolormesh(X.T, Y.T, Ep, vmin=-150, vmax=-100)
fig.colorbar(c)

fig.show()
