# test pour la création d'une matrice creuse (sparse matrix) et la résolution d'un système
# le code n'est pas écrit de manière pythonique pour des raisons didactiques, pataper!

from scipy.sparse import csc_matrix

indices_ligne = [0, 1, 2]
indices_colonne = [0, 1, 2]
valeurs = [1, 2, 3]
A = csc_matrix((valeurs, (indices_ligne, indices_colonne)), shape=(3, 3))

print(A.toarray())

import numpy
from scipy.sparse.linalg import spsolve
s = numpy.array([2, 3, 0])
print(spsolve(A, s))
