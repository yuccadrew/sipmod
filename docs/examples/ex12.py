"""
Solve 3D Poison equation for AFM CapSol model.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, condense, solve  # noqa
from sipmod import Poisson  # noqa
from docs.examples.ex11 import capsol, mesh_prefix  # noqa


def solution():
    path = os.path.dirname(mesh_prefix)
    suffix = '{:6.4f}'.format(capsol.d/capsol.Rtip)
    file = path+'/Fields.dat'+suffix
    u = np.genfromtxt(file, skip_header=1, usecols=(2))
    return u


mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='Y',
                    scale_factor=1.0)

phys = Physics(T=293.0,
               reps_w=1.0,  # relative permittivity in water
               reps_i=5.0,  # relative permittivity in solid
               c=np.array([0, 0]),  # no ions
               z=np.array([-1, 1]),
               m_w=np.array([5e-8, 5e-8]),
               s=np.array([0, 0]),  # no surface charges
               p=np.array([1, -1]),
               m_s=np.array([5e-9, 0]),
               u_0=1.0,  # probe voltage
               e_0=np.array([0, 0, 0]))  # no external electric field

A, b = Poisson.assemble(mesh, phys, aos='Y')
s, D = Poisson.dirichlet(mesh, phys)
# A, b = set_first_kind_bc(A, b, s, D)
# utils :: Solving linear system, shape=(551601, 551601).
# cg with diag preconditioner converged to tol=1e-05 and atol=0.0
# utils :: Solving done in 321.9258027076721 seconds.
# x = solve(A, b, krylov=spl.cg, pc='diag', tol=1e-05, atol=0.0)
# utils :: Implementing finshed in 0.08673381805419922 seconds.
# utils :: Solving linear system, shape=(551601, 551601).
# utils :: Solving done in 12.883355855941772 seconds.
# x = solve(A, b)
x = solve(*condense(A, b, s, D))
capsol.integrated_force(x, path=os.path.dirname(mesh_prefix))
capsol.integrated_energy(x, path=os.path.dirname(mesh_prefix))


def visualize():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    u = solution()
    # validate capsol.integrated_energy by comparing Z-U.dat and Z-U.1.dat
    capsol.integrated_energy(u, suffix='', path=os.path.dirname(mesh_prefix))
    n = capsol.r.shape[0]
    m = capsol.z.shape[0]

    i = np.arange(n)
    j = np.where(capsol.z >= 0)[0][0]
    idx = i * m + j
    ax[0, 0].plot(mesh.x[idx], u[idx], '.')
    ax[0, 0].plot(mesh.x[idx], x[idx], '.')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('X (nanometer)')
    ax[0, 0].set_title('Along Profile Y = {:4.2f}'.format(mesh.y[idx][0]))
    ax[0, 0].legend(['capsol', 'sipmod'], loc='best')

    i = np.arange(n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    ax[0, 1].plot(mesh.x[idx], u[idx], '.')
    ax[0, 1].plot(mesh.x[idx], x[idx], '.')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('X (nanometer)')
    ax[0, 1].set_title('Along Profile Y = {:4.2f}'.format(mesh.y[idx][0]))

    i = np.arange(n)
    j = np.where(capsol.z >= -capsol.d+capsol.Hwat)[0][0]
    idx = i * m + j
    ax[1, 0].plot(mesh.x[idx], u[idx], '.')
    ax[1, 0].plot(mesh.x[idx], x[idx], '.')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('X (nanometer)')
    ax[1, 0].set_title('Along Profile Y = {:4.2f}'.format(mesh.y[idx][0]))

    i = 0
    j = np.arange(m)
    idx = i * m + j
    ax[1, 1].plot(mesh.y[idx], u[idx], '.')
    ax[1, 1].plot(mesh.y[idx], x[idx], '.')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Y (nanometer)')
    ax[1, 1].set_title('Along Profile X = {:4.2f}'.format(mesh.x[idx][0]))
    plt.tight_layout()

    from sipmod.visuals import plot
    ax = plot(mesh, x)
    ax.set_xlim(-50e3, 50e3)
    ax.set_ylim(-50e3, 50e3)
    return ax


if __name__ == "__main__":
    visualize().show()
