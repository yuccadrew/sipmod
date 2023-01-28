"""
Solve 2D Poison equation with non-smooth impulse source.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import (BilinearForm, LinearForm, MeshTri, CellBasisTri,  # noqa
                    set_first_kind_bc, solve)


def solution(x, y):
    return -np.log(np.sqrt(x ** 2 + y ** 2)) / (2 * np.pi)


def source(x, y):
    dist = np.sqrt(x ** 2 + y ** 2)
    f = np.zeros_like(dist)
    f[dist == 0] = 1.0
    return f


@BilinearForm
def laplace(u, v, w):
    return (
        w.c * u.grad[0] * v.grad[0] +
        w.c * u.grad[1] * v.grad[1]
    ) * w.area


@LinearForm
def impulse_load(_, w):
    return (
        w.f[0] * w.kron[0] +
        w.f[1] * w.kron[1] +
        w.f[2] * w.kron[2]
    ) / (2 * np.pi)


mesh_prefix = 'docs/examples/meshes/mesh_ex01'
mesh = MeshTri.read(mesh_prefix+'.1', scale_factor=1e-6)
basis = CellBasisTri(mesh)

f = source(basis.xverts, basis.yverts)
A = laplace(c=1.0).assemble(basis)
b = impulse_load(f=f).assemble(basis)

boundary_nodes = mesh.boundary_nodes('outer')
D = boundary_nodes['id']
s = solution(boundary_nodes['x'], boundary_nodes['y'])
A, b = set_first_kind_bc(A, b, s, D)
x = solve(A, b)


def demo():
    from sipmod.visuals import plot
    v = basis.interpolate(x)['value']
    return plot(mesh, v, cmap='jet', vmax=0.6)


def visualize():
    import matplotlib.pyplot as plt
    dist = np.sqrt(basis.x**2 + basis.y**2)
    u = solution(basis.x, basis.y)
    v = basis.interpolate(x)['value']
    plt.plot(dist, u, '.')
    plt.plot(dist, v, '.')

    from sipmod.visuals import plot
    return plot(mesh, v, cmap='jet', vmax=0.6)


if __name__ == "__main__":
    visualize().show()
