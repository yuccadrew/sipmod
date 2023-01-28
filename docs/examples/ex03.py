"""
Solve 3D Maxwell-Wagner Polarization for spherical particles.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from scipy.constants import epsilon_0  # noqa
from sipmod import (BilinearForm, LinearForm, MeshTri, CellBasisTri,  # noqa
                    set_first_kind_bc, solve)


cond_i = 4.5j * 1e7 * epsilon_0  # particle conductivity
cond_w = 80j * 1e7 * epsilon_0 + 0.05  # electrolyte conductivity
R = 0.1e-6  # radius of the spherical particle


def solution(x, y):
    dist = np.sqrt(x ** 2 + y ** 2)
    u = np.zeros(dist.shape[0], dtype=complex)
    ratio = (cond_i - cond_w) / (2 * cond_w + cond_i)
    mask = dist >= R
    u[mask] = (ratio * (R / dist[mask])**3 - 1) * x[mask]
    u[~mask] = -3 * cond_w / (2 * cond_w + cond_i) * x[~mask]
    return u


@BilinearForm(dtype=np.complex128, aos='x')
def laplace(u, v, w):
    return (
        w.c * u.grad[0] * v.grad[0] +
        w.c * u.grad[1] * v.grad[1]
    ) * w.area * w.r


mesh_prefix = 'docs/examples/meshes/mesh_ex03'
mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='X',
                    scale_factor=1e-6)
ibasis = CellBasisTri(mesh, 'solid')
wbasis = CellBasisTri(mesh, 'water')

A = (
    laplace.assemble(ibasis, c=cond_i) +
    laplace.assemble(wbasis, c=cond_w)
)
b = np.zeros(A.shape[0])

boundary_nodes = mesh.boundary_nodes('outer')
D = boundary_nodes['id']
s = -1.0 * boundary_nodes['x']
A, b = set_first_kind_bc(A, b, s, D)
x = solve(A, b)


def demo():
    import matplotlib.pyplot as plt
    from sipmod.visuals import plot
    fig, ax = plt.subplots(2, 1, figsize=(6, 5*2))
    plot(mesh, x.real, edgecolors='gray', contours=True, ax=ax[0])
    ax[0].set_title('Real')
    plot(mesh, x.imag, edgecolors='gray', contours=True, ax=ax[1])
    ax[1].set_title('Imag')
    return plt


def visualize():
    import matplotlib.pyplot as plt
    dist = np.sqrt(mesh.x ** 2 + mesh.y ** 2)
    u = solution(mesh.x, mesh.y)
    fig, ax = plt.subplots()
    plt.plot(dist, u.real, '.')
    plt.plot(dist, x.real, '.')
    plt.plot(dist, np.abs(u.real - x.real), '.')

    fig, ax = plt.subplots()
    plt.plot(dist, u.imag, '.')
    plt.plot(dist, x.imag, '.')
    plt.plot(dist, np.abs(u.imag - x.imag), '.')

    from sipmod.visuals import plot
    plot(mesh, x.real, edgecolors='gray', contours=True)
    plot(mesh, x.imag, edgecolors='gray', contours=True)
    return plt


if __name__ == "__main__":
    visualize().show()
