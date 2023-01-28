"""
Solve 3D Poisson Boltzmann equation for spherical particle in water and solid
using PoissonBoltzmann class.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, set_first_kind_bc, solve  # noqa
from sipmod import PoissonBoltzmann as PB  # noqa


def solution(p: Physics, x, y):
    u = np.zeros(x.shape[0])
    zeta = - p.sigma_w / (p.kappa[0] + 1 / p.R) / p.eps_w
    dist = np.sqrt(x ** 2 + y ** 2)
    mask = dist > p.R
    u[mask] = zeta * p.R / dist[mask] * np.exp((p.R - dist[mask]) * p.kappa[0])
    u[~mask] = zeta
    return u


phys = Physics(T=293., reps_w=80., reps_i=4.5,
               c=np.array([1, 1]),
               z=np.array([-1, 1]),
               m_w=np.array([5e-8, 5e-8]),
               s=np.array([0., 0.01]),
               p=np.array([1, -1]),
               m_s=np.array([5e-9, 0]),
               R=0.1e-6,
               water_only=False)  # use either True or False for this case

mesh_prefix = 'docs/examples/meshes/mesh_ex03'
mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='X',
                    scale_factor=1e-6)

form_args = {'aos': 'x',
             'phys': phys,
             'prev': np.zeros(mesh.nnodes)}

A, b = PB.assemble(mesh, phys, **form_args)
s, D = PB.dirichlet(mesh, phys, **form_args)
A, b = set_first_kind_bc(A, b, s, D)
x = solve(A, b)


def demo():
    from sipmod.visuals import plot
    ax = plot(mesh, x, cmap='viridis')
    return ax


def visualize():
    import matplotlib.pyplot as plt
    dist = np.sqrt(mesh.x ** 2 + mesh.y ** 2)
    u = solution(phys, mesh.x, mesh.y)
    fig, ax = plt.subplots()
    ax.plot(dist, u, '.')
    ax.plot(dist, x, '.')
    ax.set_xlim([0.9e-7, 1.5e-7])

    from sipmod.visuals import plot
    ax = plot(mesh, x, cmap='viridis')
    return ax


if __name__ == "__main__":
    visualize().show()
