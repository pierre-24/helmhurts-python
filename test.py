import numpy

from helmhurts.solver_system import HelmholtzSolverSystem

import matplotlib.pyplot as plt

sz = (256, 256)

# diffraction
n = numpy.ones(sz, dtype=complex)
n[100:110, 5:] = 2.24 - 0.021j

# source
f = numpy.zeros(sz)
f[10:20, 10:20] = 1

solver = HelmholtzSolverSystem(52, n, f, (.01, .01))
E = solver.E()

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

c = axes.pcolormesh(X.T, Y.T, Ep, vmin=-150, vmax=numpy.max(Ep))
fig.colorbar(c)

fig.show()