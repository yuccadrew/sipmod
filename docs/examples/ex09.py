"""
Solve perturbation part of 3D Poisson Nernst Planck equations for double-layer
polarization using PoissonNernstPlanck class in water and solid.
"""

import sys
import os
sys.path.append(os.getcwd())

import logging  # noqa
import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, set_first_kind_bc, solve  # noqa
from sipmod import PoissonNernstPlanck as PNP  # noqa


logging.basicConfig(level=logging.INFO, format='%(module)s :: %(message)s')


def static_solution(p: Physics, x, y):
    u = np.zeros(x.shape[0])
    zeta = - p.sigma_w / (p.kappa[0] + 1 / p.R) / p.eps_w
    dist = np.sqrt(x ** 2 + y ** 2)
    mask = dist > p.R
    u[mask] = zeta * p.R / dist[mask] * np.exp((p.R - dist[mask]) * p.kappa[0])
    u[~mask] = zeta
    return u


def cartesian2spherical(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # radial distance
    theta = np.arccos(z / rho)  # polar angle
    # phi = np.arctan2(y / x)
    phi = np.zeros_like(rho) + np.pi / 2  # azimuthal angle
    mask = x > 0
    phi[mask] = np.arctan(y[mask] / x[mask])
    mask = x < 0
    phi[mask] = np.arctan(y[mask] / x[mask]) + np.pi
    return rho, theta, phi


phys = Physics(T=293., reps_w=80., reps_i=4.5,
               c=np.array([1, 1]),
               z=np.array([-1, 1]),
               m_w=np.array([5e-8, 5e-8]),
               s=np.array([0.01*0.8, 0.01]),  # ratio 0.8
               p=np.array([1, -1]),
               m_s=np.array([5e-9, 0]),
               R=5e-6,
               e_0=np.array([100., 0, 0]),
               perfect_conductor=False,  # must be False for this case
               steady_state=False,  # must be False for this case
               eliminate_sigma=False,  # must be False for this case
               water_only=False)  # should be False if dpot is nonzero in solid

mesh_prefix = 'docs/examples/meshes/mesh_ex04'
mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='X',
                    scale_factor=1e-6)

dof = 4  # must be 4 in this case
# scale_factor = 1 / np.hstack([phys.D_w, [phys.eps_w, phys.D_s[0]]])
prev = np.zeros((mesh.nnodes, dof), dtype=np.complex128)
# prev[:, 2] = static_solution(phys, mesh.x, mesh.y)
prev[:, 2] = np.load('docs/examples/meshes/sphere_s62_static_cinf_1.npy')
form_args = {'dtype': np.complex128,
             'aos': 'x',
             'dof': dof,
             'prev': prev.ravel(),
             'phys': phys,
             'freq': 100/(2*np.pi),
             'scale_factor': None}

A, b = PNP.assemble(mesh, phys, **form_args)
s, D = PNP.dirichlet(mesh, phys, **form_args)
A, b = set_first_kind_bc(A, b, s, D)
x = solve(A, b)


def demo():
    import matplotlib.pyplot as plt
    from sipmod.visuals import plot
    fig, ax = plt.subplots(2, 1, figsize=(6, 5*2))
    plot(mesh, x.reshape(-1, dof)[:, 2].real, cmap='viridis', ax=ax[0])
    ax[0].set_title('Real')
    plot(mesh, x.reshape(-1, dof)[:, 2].imag, cmap='viridis', ax=ax[1])
    ax[1].set_title('Imag')
    return plt


def visualize():
    import matplotlib.pyplot as plt
    boundary_nodes = mesh.boundary_nodes('stern')
    nid = boundary_nodes['id']
    angle = cartesian2spherical(
        boundary_nodes['x'],
        boundary_nodes['y'],
        np.zeros_like(boundary_nodes['x'])
    )[2] * 180 / np.pi
    fig, ax = plt.subplots(2, 4, figsize=(8, 4.5))
    ax[0][0].plot(angle, x.reshape(-1, dof)[nid, 0].real, '.', markersize=.1)
    ax[0][1].plot(angle, x.reshape(-1, dof)[nid, 1].real, '.', markersize=.1)
    ax[0][2].plot(angle, x.reshape(-1, dof)[nid, 2].real, '.', markersize=.1)
    ax[0][3].plot(angle, x.reshape(-1, dof)[nid, 3].real, '.', markersize=.1)

    ax[1][0].plot(angle, x.reshape(-1, dof)[nid, 0].imag, '.', markersize=.1)
    ax[1][1].plot(angle, x.reshape(-1, dof)[nid, 1].imag, '.', markersize=.1)
    ax[1][2].plot(angle, x.reshape(-1, dof)[nid, 2].imag, '.', markersize=.1)
    ax[1][3].plot(angle, x.reshape(-1, dof)[nid, 3].imag, '.', markersize=.1)
    plt.tight_layout()

    from sipmod.visuals import plot
    ax = plot(mesh, x.reshape(-1, dof)[:, 2].real, cmap='viridis')
    ax = plot(mesh, x.reshape(-1, dof)[:, 2].imag, cmap='viridis')
    return ax


if __name__ == "__main__":
    visualize().show()
