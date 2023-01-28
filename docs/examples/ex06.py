"""
Solve 3D Poisson Nernst Planck equations for diffuse-layer polarization
with four unknowns per node.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
from scipy import constants  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, set_first_kind_bc, solve  # noqa
from sipmod import PoissonNernstPlanck as PNP  # noqa


def solution(p: Physics, x, y, freq, alpha=0, beta=0):
    # modified spherical Bessel function of the second kind
    def k_1(k, r):
        kr = k * r
        return np.pi / 2 * np.exp(-kr) * (1 / kr + 1 / kr ** 2)

    # derivative of the modified spherical Bessel function of the second kind
    # with respect to r evaluated at r=a
    def g_1(k, r):
        kr = k * r
        return (
            -k * k_1(k, r) -
            np.pi / 2 * np.exp(-kr) / r * (1 / kr + 2 / kr ** 2)
        )

    u = np.zeros((x.shape[0], 4), dtype=np.complex128)
    cneg = u[:, 0]
    cpos = u[:, 1]
    pot = u[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2)

    P = p.eps_w  # permittivity of electrolyte
    M = p.m_w[0]  # mobility in electrolyte
    D = p.D_w[0]  # diffusion coefficient in electrolyte
    R = p.R  # radius of spherical particle
    L = 1 / p.kappa[0]  # Debye length
    E = np.linalg.norm(p.e_0)  # amplitude of electrical field
    F = constants.e * constants.N_A

    c_1 = p.c[0]
    c_2 = p.c[1]
    c_3 = 0.0
    d_1 = np.sqrt(2j * np.pi * freq / D + 1 / L ** 2)
    d_2 = np.sqrt(2j * np.pi * freq / D)

    a_1 = d_1 * R
    a_2 = d_2 * R
    f_2 = (a_1 ** 2 + 2 * a_1 + 2) / (a_1 + 1)
    f_3 = (a_2 + 1) / (a_2 ** 2 + 2 * a_2 + 2)
    f_1 = f_2 * 2j * np.pi * freq / D * L ** 2

    numerator = (
        3 * (1 + beta * R / D * f_3) +
        3 * c_3 / (c_3 - 2 * c_1) * (alpha / M - 1)
    )
    denominator = (
        c_3 / (c_3 - 2 * c_1) * (
            f_1 + alpha / M * (f_2 - 2) +
            beta * R * d_1 ** 2 / D * L ** 2 + 2
        ) - (2 + f_1) * (1 + beta * R / D * f_3)
    )

    k_e = E * (1 + numerator / denominator)
    f_e = f_2 + (E + 2 * k_e) / (E - k_e)
    k_a = (-E * R + k_e * R) * d_1 ** 2 * P / 2 / F / k_1(d_1, R)
    k_b = (E - k_e) / g_1(d_2, R) * (
        d_1 ** 2 * P / 2 / F * f_2 - M / D * c_1 * f_e
    )
    k_m = -c_3 / c_1 * (E - k_e) / g_1(d_2, R) / (1 + beta * R / D * f_3) * (
        d_1 ** 2 * P / 2 / F * (f_2 + beta * R / D) -
        c_1 / D * (M - alpha) * f_e
    )

    mask = dist >= R
    rho = dist[mask]
    cosb = x[mask] / dist[mask]
    k_11 = k_1(d_1, rho)
    k_12 = k_1(d_2, rho)
    cneg[mask] = -(k_a * k_11 + k_b * k_12) * cosb
    cpos[mask] = (c_2 / c_1 * k_a * k_11 + (k_b - k_m) * k_12) * cosb
    pot[mask] = (-2 * F / d_1 ** 2 / P * k_a * k_11 -
                 E * rho + k_e * R ** 3 / rho ** 2) * cosb
    return u


phys = Physics(T=293., reps_w=80., reps_i=4.5,
               c=np.array([1, 1]),
               z=np.array([-1, 1]),
               m_w=np.array([5e-8, 5e-8]),
               s=np.array([0., 0.01]),
               p=np.array([1, -1]),
               m_s=np.array([5e-9, 0]),
               R=0.1e-6,
               e_0=np.array([1., 0., 0.]),
               perfect_conductor=True,  # must be True for this case
               steady_state=False,  # must be False for this case
               eliminate_sigma=False,  # use either T or F if PEC is True
               water_only=False)  # use either True or False for this case

mesh_prefix = 'docs/examples/meshes/mesh_ex03'
mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='X',
                    scale_factor=1e-6)

dof = 4  # use either 3 or 4 for this case
form_args = {'dtype': np.complex128,
             'aos': 'x',
             'dof': dof,
             'prev': np.zeros(mesh.nnodes * dof),
             'phys': phys,
             'freq': 3e4/(2*np.pi)}

A, b = PNP.assemble(mesh, phys, **form_args)
s, D = PNP.dirichlet(mesh, phys, **form_args)
A, b = set_first_kind_bc(A, b, s, D)
x = solve(A, b)


def demo():
    import matplotlib.pyplot as plt
    from sipmod.visuals import plot
    titles = [r'$\delta C_-$', r'$\delta C_+$', r'$\delta U$']
    units = [r'$(\mu mol/m^3)$', r'$(\mu mol/m^3)$', r'$(\mu V)$']
    vmin = np.array([2.5e-6, 2.5e-6, 0.2e-6]) * 1e6 * (-1)
    vmax = np.array([2.5e-6, 2.5e-6, 0.2e-6]) * 1e6
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(6, 6))
    for i in range(3):
        plot(mesh, x.reshape(-1, dof)[:, i].real*1e6, cmap='YlGnBu_r',
             ax=ax[i][0], vmin=vmin[i], vmax=vmax[i],
             scale_factor=1e6)
        ax[i][0].set_aspect('equal')
        ax[i][0].set_xlim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][0].set_ylim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][0].set_xlabel(r'X ($\mu m$)')
        ax[i][0].set_ylabel(r'Y ($\mu m$)')
        ax[i][0].set_title('Real ' + titles[i])

        plot(mesh, x.reshape(-1, dof)[:, i].imag*1e6, cmap='YlGnBu_r',
             ax=ax[i][1], vmin=vmin[i]*1e-2, vmax=vmax[i]*1e-2,
             scale_factor=1e6, cbtitle=units[i])
        ax[i][1].set_aspect('equal')
        ax[i][1].set_xlim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][1].set_ylim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][1].set_xlabel(r'X ($\mu m$)')
        ax[i][1].set_ylabel(r'Y ($\mu m$)')
        ax[i][1].set_title('Imag ' + titles[i])
    plt.tight_layout()
    return plt


def visualize():
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn-poster')
    titles = [r'$\delta C_-$', r'$\delta C_+$', r'$\delta U$']
    units = [r'$(\mu mol/m^3)$', r'$(\mu mol/m^3)$', r'$(\mu V)$']

    boundary_nodes = mesh.boundary_nodes('aos')
    mask = boundary_nodes['x'] > phys.R
    nid = boundary_nodes['id'][mask]
    dist = (np.sqrt(mesh.x ** 2 + mesh.y ** 2) - phys.R) / phys.debye_length[0]
    u = solution(phys, mesh.x, mesh.y, freq=3e4/(2*np.pi))
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(6, 6))
    for i in range(3):
        ax[i][0].plot(dist[nid], u.reshape(-1, 4)[nid, i].real * 1e6, '.')
        ax[i][0].plot(dist[nid], x.reshape(-1, dof)[nid, i].real * 1e6, '.')
        ax[i][0].set_xscale('log')
        ax[i][0].set_ylabel(units[i])
        ax[i][0].set_title('Real ' + titles[i])

        ax[i][1].plot(dist[nid], u.reshape(-1, 4)[nid, i].imag * 1e6, '.')
        ax[i][1].plot(dist[nid], x.reshape(-1, dof)[nid, i].imag * 1e6, '.')
        ax[i][1].set_xscale('log')
        ax[i][1].set_title('Imag ' + titles[i])
    ax[2][0].set_xlabel('Normalized Distance')
    ax[2][1].set_xlabel('Normalized Distance')
    ax[0][1].legend(['analytical', 'numerical'])
    plt.tight_layout()

    from sipmod.visuals import plot
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(6, 6))
    vmin = np.array([2.5e-6, 2.5e-6, 0.2e-6]) * 1e6 * (-1)
    vmax = np.array([2.5e-6, 2.5e-6, 0.2e-6]) * 1e6
    for i in range(3):
        plot(mesh, x.reshape(-1, dof)[:, i].real*1e6, cmap='YlGnBu_r',
             ax=ax[i][0], vmin=vmin[i], vmax=vmax[i],
             scale_factor=1e6)
        ax[i][0].set_aspect('equal')
        ax[i][0].set_xlim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][0].set_ylim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][0].set_xlabel(r'X ($\mu m$)')
        ax[i][0].set_ylabel(r'Y ($\mu m$)')
        ax[i][0].set_title('Real ' + titles[i])

        plot(mesh, x.reshape(-1, dof)[:, i].imag*1e6, cmap='YlGnBu_r',
             ax=ax[i][1], vmin=vmin[i]*1e-2, vmax=vmax[i]*1e-2,
             scale_factor=1e6, cbtitle=units[i])
        ax[i][1].set_aspect('equal')
        ax[i][1].set_xlim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][1].set_ylim(-2 * phys.R * 1e6, 2 * phys.R * 1e6)
        ax[i][1].set_xlabel(r'X ($\mu m$)')
        ax[i][1].set_ylabel(r'Y ($\mu m$)')
        ax[i][1].set_title('Imag ' + titles[i])
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    visualize().show()
