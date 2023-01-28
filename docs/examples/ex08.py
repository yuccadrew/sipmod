"""
Solve static part of 3D Poisson Nernst Planck equations for double-layer
polarization using PoissonNernstPlanck class in water and solid.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, set_first_kind_bc, solve  # noqa
from sipmod import PoissonNernstPlanck as PNP  # noqa


def static_solution(p: Physics, x, y):
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
               s=np.array([0.01*0.8, 0.01]),  # ratio 0.8
               p=np.array([1, -1]),
               m_s=np.array([5e-9, 0]),
               R=5e-6,
               e_0=np.array([100., 0, 0]),
               perfect_conductor=False,  # must be False for this case
               steady_state=True,  # must be True for this case
               eliminate_sigma=True,  # must be True for this case
               water_only=False)  # use either True or False for this case

mesh_prefix = 'docs/examples/meshes/mesh_ex04'
mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='X',
                    scale_factor=1e-6)

dof = 4  # use either 3 or 4 for this case
# scale_factor = 1 / np.hstack([phys.D_w, [phys.eps_w, phys.D_s[0]]])
form_args = {'dtype': np.float64,
             'aos': 'x',
             'dof': dof,
             'prev': np.zeros(mesh.nnodes * dof),
             'phys': phys,
             'freq': None,
             'scale_factor': None}

A, b = PNP.assemble(mesh, phys, **form_args)
s, D = PNP.dirichlet(mesh, phys, **form_args)
A, b = set_first_kind_bc(A, b, s, D)
x = solve(A, b)


def demo():
    from sipmod.visuals import plot
    ax = plot(mesh, x.reshape(-1, dof)[:, 2].real, cmap='viridis')
    return ax


def visualize():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-poster')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    dist = np.sqrt(mesh.x ** 2 + mesh.y ** 2)
    u = np.load('docs/examples/meshes/sphere_s62_static_cinf_1.npy').real
    boundary_nodes = mesh.boundary_nodes('aos')
    mask = ((boundary_nodes['x'] >= phys.R) &
            (boundary_nodes['x'] < phys.R + 10 * phys.debye_length[0]))
    nid = boundary_nodes['id'][mask]
    ax[0].plot(dist[nid], x.reshape(-1, dof)[nid, 2], '.')
    ax[0].plot(dist[nid], u[nid], '.')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
    dist = np.sqrt(mesh.x ** 2 + mesh.y ** 2)
    u = static_solution(phys, mesh.x, mesh.y)
    mask = ((dist >= 0.9 * phys.R) &
            (dist <= phys.R + 20 * phys.debye_length[0]))
    nid = np.where(mask)[0]
    ax[0].plot(dist[nid], u[nid], '.', alpha=0.1)
    ax[0].plot(dist[nid], x.reshape(-1, dof)[nid, 2], '.', markersize=5)
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_title('$U^{(0)}_a$')
    boundary_nodes = mesh.boundary_nodes('aos')
    mask = ((boundary_nodes['x'] >= phys.R) &
            (boundary_nodes['x'] <= phys.R + 4 * phys.debye_length[0]))
    nid = boundary_nodes['id'][mask]
    ax[1].plot(dist[nid], x.reshape(-1, dof)[nid, 2], '.')
    ax[1].plot(dist[nid], u[nid], '.')
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_title('$U^{(0)}_a$')
    plt.tight_layout()

    from sipmod.visuals import plot
    plt.style.use('default')
    ax = plot(mesh, x.reshape(-1, dof)[:, 2].real, cmap='viridis')
    return ax


if __name__ == "__main__":
    visualize().show()
