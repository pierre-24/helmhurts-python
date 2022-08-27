import numpy
from scipy import sparse
from scipy.sparse.linalg import spsolve as sparse_solve

import matplotlib.pyplot as plt

M = 128
N = 128
delta_x, delta_y = 2. / M, 2. / N
idelta2x, idelta2y = delta_x ** -2, delta_y ** -2
k = 9
boundary_left, boundary_right, boundary_top, boundary_bottom = 0,0, 1, 1

n_sparse_value = 5 * M * N + 2 * (M+2) + 2 * (N+2)
sparse_array_index = numpy.zeros((n_sparse_value, 2), dtype=int)  # x, y, value
sparse_array_value = numpy.zeros(n_sparse_value)
i = 0

print(n_sparse_value, '{:.3f} %'.format(n_sparse_value / (((M+2)*(N+2)))**2 * 100))

# define the system of equation
print('create main system')
for x in range(1, M + 1):
    for y in range(1, N + 1):

        index_0 = numpy.ravel_multi_index((x, y), (M + 2, N + 2))
        sparse_array_index[i] = index_0, index_0
        sparse_array_value[i] = k ** 2 - 2 * idelta2x - 2 * idelta2y

        xm = x - 1
        xp = x + 1
        ym = y - 1
        yp = y + 1
        i += 1

        for coo in [(xm, y, idelta2x), (xp, y, idelta2x), (x, ym, idelta2y), (x, yp, idelta2y)]:
            index = numpy.ravel_multi_index(coo[:2], (M + 2, N + 2))
            sparse_array_index[i] = index_0, index
            sparse_array_value[i] = coo[2]
            i += 1

# define the source
print('define source')
length = (M + 2) * (N + 2)
f = numpy.zeros(length)
f[numpy.ravel_multi_index((int(M/2), int(N/2)), (M+2, N+2))] = 1

# add the additional equation to set dirichlet boundary conditions
print('add boundary')
for j in range(0, M+2):
    for coo in [(j, 0, boundary_top), (j, N+1, boundary_bottom)]:
        index = numpy.ravel_multi_index(coo[:2], (M+2, N+2))
        f[index] = coo[2]
        sparse_array_index[i] = index, index
        sparse_array_value[i] = 1
        i += 1

for j in range(1, N+1):
    for coo in [(0, j, boundary_left), (M+1, j, boundary_right)]:
        index = numpy.ravel_multi_index(coo[:2], (M+2, N+2))
        f[index] = coo[2]
        sparse_array_index[i] = index, index
        sparse_array_value[i] = 1
        i += 1

# compute the thing
print('create S')
S = sparse.csc_array((sparse_array_value, (sparse_array_index[:, 0], sparse_array_index[:, 1])), shape=(length, length))

print('solve the system')
E = sparse_solve(S, f).reshape((M+2, N+2))
del S

# plot it
fig = plt.figure(figsize=(10, 8))
axes = fig.add_subplot()
axes.axis([-1, 1, -1, 1])

ve = numpy.max([-numpy.min(E), numpy.max(E)])

b, a = numpy.meshgrid(numpy.linspace(-1, 1, M), numpy.linspace(-1, 1, N))
c = axes.pcolormesh(a, b, E[1:M+1, 1:N+1], vmin=-ve, vmax=ve)
fig.colorbar(c)

plt.show()
