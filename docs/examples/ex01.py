"""
Solve 2D Poison equation with smooth source evaluated in elemental centers.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import (BilinearForm, LinearForm, MeshTri, CellBasisTri,  # noqa
                    set_first_kind_bc, solve)


def solution(x, y):
    return np.cos(np.pi * np.sqrt(x ** 2 + y ** 2) / 2)


@BilinearForm
def laplace(u, v, w):
    c = 1.
    return (
        c * u.grad[0] * v.grad[0] +
        c * u.grad[1] * v.grad[1]
    ) * w.area


@LinearForm
def nonuniform_load(_, w):
    dist = np.sqrt(w.x ** 2 + w.y ** 2)
    f = np.zeros_like(dist)
    mask = dist > 0
    f[mask] = (np.sin(np.pi / 2 * dist[mask]) / dist[mask] +
               np.cos(np.pi / 2 * dist[mask]) * np.pi / 2) * np.pi / 2
    f[~mask] = np.pi ** 2 / 2
    return (f / 3) * w.area


mesh_prefix = 'docs/examples/meshes/mesh_ex01'
mesh = MeshTri.read(mesh_prefix+'.1', scale_factor=1e-6)
basis = CellBasisTri(mesh)

A = laplace.assemble(basis)
b = nonuniform_load.assemble(basis)

boundary_nodes = mesh.boundary_nodes('outer')
D = boundary_nodes['id']
s = solution(boundary_nodes['x'], boundary_nodes['y'])
A, b = set_first_kind_bc(A, b, s, D)
# x = solve(A, b, spl.cg, pc='ilu', tol=1e-5, atol=0)
x = solve(A, b)


def demo():
    from sipmod.visuals import plot
    return plot(mesh, x, cmap='jet')


def visualize():
    import matplotlib.pyplot as plt
    dist = np.sqrt(mesh.x ** 2 + mesh.y ** 2)
    u = solution(mesh.x, mesh.y)
    plt.plot(dist, u, '.')
    plt.plot(dist, x, '.')

    from sipmod.visuals import plot
    return plot(mesh, x, cmap='jet')


if __name__ == "__main__":
    visualize().show()
