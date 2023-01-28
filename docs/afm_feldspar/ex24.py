"""
Solve 3D Poisson Nernst Planck equations for AFM feldspar model with 98 nm
tip-sample separation. Use capsol.post_processing to process solution. Fix
sigma at the solid-water interface.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, FourierDLF, condense, solve  # noqa
from sipmod import PoissonNernstPlanck as PNP  # noqa
from docs.afm_feldspar.ex23 import capsol, mesh_prefix  # noqa


mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='Y',
                    scale_factor=1e-9)

# https://gpg.geosci.xyz/content/physical_properties/tables/permittivity_minerals.html
phys = Physics(T=293.0,
               reps_w=80.0,  # relative permittivity in water
               reps_i=6.0,  # relative permittivity in solid
               c=np.array([0, 0]),  # no ions
               z=np.array([-2, 2]),  # for calcite model
               m_w=np.array([5e-15, 5e-15]),  # ?
               s=np.array([0.29, 0.29]),  # 1.8 e/nm^2
               p=np.array([1, -1]),  # !!! take care for decoupled system
               m_s=np.array([4e-14, 0]),  # lead to diffusivity as 1e-15 m^2/s
               u_0=10.0,  # probe voltage
               e_0=np.array([0, 0, 0]))  # no external electric field

dlf = FourierDLF(time=np.logspace(-6, 8, 301))


def run():
    from tqdm import tqdm
    from sipmod import CellBasisTri, FacetBasisTri
    from sipmod.utils import load_dict, amend_dict
    prev_mesh = MeshTri.read('docs/afm_feldspar/meshes/mesh_ex21.1',
                             axis_of_symmetry='Y',
                             scale_factor=1e-9)
    dof = 4  # must be 4 in this case
    indices = mesh.boundary_nodes('stern')['id'] * dof + (dof - 1)
    prev_indices = prev_mesh.boundary_nodes('stern')['id'] * dof + (dof - 1)
    prev_dcEM = load_dict('docs/afm_feldspar/meshes/ex22t0.h5', 'dcEM')
    prev_fEM = load_dict('docs/afm_feldspar/meshes/ex22t0.h5', 'fEM')

    path = os.path.dirname(mesh_prefix)
    basis = {'water': CellBasisTri(mesh, 'water'),
             'solid': CellBasisTri(mesh, 'solid'),
             'air': CellBasisTri(mesh, 'air'),
             'stern': FacetBasisTri(mesh, 'stern')}
    form_args = {'dtype': np.complex128,
                 'aos': 'Y',
                 'dof': dof,
                 'prev': np.zeros(mesh.nnodes*dof),
                 'freq': 0.0,
                 'scale_factor': None,
                 'basis': basis,
                 's': prev_dcEM[prev_indices],
                 'D': indices}
    A, b = PNP.assemble(mesh, phys, **form_args)
    s, D = PNP.dirichlet(mesh, phys, **form_args)
    x = solve(*condense(A, b, s, D))
    pot = x.reshape(-1, dof)[:, 2]
    Er, Ez, Fz = capsol.post_processing(pot.real, path=path)
    amend_dict(path+'/ex24.h5', data={'dcEM': x,
                                      'dcEr': Er,
                                      'dcEz': Ez,
                                      'dcFz': Fz})

    x = np.zeros((dlf.freq.shape[0], mesh.nnodes*dof), dtype=np.complex128)
    Er = np.zeros((dlf.freq.shape[0], mesh.nnodes))
    Ez = np.zeros((dlf.freq.shape[0], mesh.nnodes))
    Fz = np.zeros((dlf.freq.shape[0], 8))
    amend_dict(path+'/ex24.h5', data={'fEM': x,
                                      'fEr': Er,
                                      'fEz': Ez,
                                      'fFz': Fz})

    for i in tqdm(range(dlf.freq.shape[0])):
        print('Calculating frequency {:4.2e} Hz'.format(dlf.freq[i]))
        form_args['freq'] = dlf.freq[i]
        form_args['s'] = prev_fEM[i, prev_indices]
        A, b = PNP.assemble(mesh, phys, **form_args)
        s, D = PNP.dirichlet(mesh, phys, **form_args)
        x[i, :] = solve(*condense(A, b, s, D))
        pot = x[i, :].reshape(-1, dof)[:, 2]
        Er, Ez, Fz = capsol.post_processing(pot.real, path=path)
        amend_dict(path+'/ex24.h5',
                   data={'fEM': x[i, :], 'fEr': Er, 'fEz': Ez, 'fFz': Fz},
                   replace_row=i)
    return x


def validate():
    import matplotlib.pyplot as plt
    from sipmod.utils import load_dict
    prev_mesh = MeshTri.read('docs/afm_feldspar/meshes/mesh_ex21.1',
                             axis_of_symmetry='Y',
                             scale_factor=1e-9)

    path = os.path.dirname(mesh_prefix)
    prev_indices = prev_mesh.boundary_nodes('stern')['id'] * 4 + 3
    indices = mesh.boundary_nodes('stern')['id'] * 4 + 3
    prev_u = load_dict('docs/afm_feldspar/meshes/ex22t0.h5', 'dcEM')
    # prev_u = load_dict('docs/afm_feldspar/meshes/ex22t0.h5', 'fEM')[100, :]
    u = load_dict(path+'/ex24.h5', 'dcEM')
    # u = load_dict(path+'/ex24.h5', 'fEM')[100, :]
    np.testing.assert_array_equal(prev_indices+2*4, indices)
    np.testing.assert_allclose(prev_u[prev_indices], u[indices])

    pot = u.reshape(-1, 4)[:, 2]
    sigma = u.reshape(-1, 4)[:, 3]
    n = capsol.r.shape[0]
    m = capsol.z.shape[0]

    # display potential and surface charge density at solid-water interface
    fig, ax = plt.subplots(2, 2)
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    ax[0, 0].plot(mesh.x[idx], pot[idx].real, '.', color='tab:blue')
    ax[0, 0].plot(-mesh.x[idx], pot[idx].real, '.', color='tab:blue')
    ax[0, 0].set_xlim(-1e-6, 1e-6)
    ax[0, 0].set_xlabel('X (m)')
    ax[0, 0].set_ylabel(r'$\phi$ (V)')
    ax[0, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[0, 1].plot(mesh.x[idx], sigma[idx].real, '.', color='tab:orange')
    ax[0, 1].plot(-mesh.x[idx], sigma[idx].real, '.', color='tab:orange')
    ax[0, 1].set_xlim(-1e-6, 1e-6)
    ax[0, 1].set_xlabel('X (m)')
    ax[0, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[0, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 0].plot(mesh.x[idx], pot[idx].real, '.', color='tab:blue')
    ax[1, 0].plot(-mesh.x[idx], pot[idx].real, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('X (m)')
    ax[1, 0].set_ylabel(r'$\phi$ (V)')
    ax[1, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 1].plot(mesh.x[idx], sigma[idx].real, '.', color='tab:orange')
    ax[1, 1].plot(-mesh.x[idx], sigma[idx].real, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('X (m)')
    ax[1, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[1, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))
    plt.tight_layout()

    from sipmod.visuals import plot
    ax = plot(mesh, pot.real)
    ax.set_xlim(-50e-6, 50e-6)
    ax.set_ylim(-50e-6, 50e-6)
    return ax


def visualize():
    import matplotlib.pyplot as plt
    from sipmod.utils import load_dict
    prev_dcFz = load_dict('docs/afm_feldspar/meshes/ex22t0.h5', 'dcFz')
    prev_fFz = load_dict('docs/afm_feldspar/meshes/ex22t0.h5', 'fFz')

    path = os.path.dirname(mesh_prefix)
    dcEM = load_dict(path+'/ex24.h5', 'dcEM')
    dcFz = load_dict(path+'/ex24.h5', 'dcFz')
    fEM = load_dict(path+'/ex24.h5', 'fEM')
    fFz = load_dict(path+'/ex24.h5', 'fFz')
    fPot = fEM.reshape(-1, mesh.nnodes, 4)[:, :, 2]
    fSigma = fEM.reshape(-1, mesh.nnodes, 4)[:, :, 3]
    fCneg = fEM.reshape(-1, mesh.nnodes, 4)[:, :, 0]
    fCpos = fEM.reshape(-1, mesh.nnodes, 4)[:, :, 1]
    dcPot = dcEM.reshape(-1, 4)[:, 2]
    dcSigma = dcEM.reshape(-1, 4)[:, 3]
    dcCneg = dcEM.reshape(-1, 4)[:, 0]
    dcCpos = dcEM.reshape(-1, 4)[:, 1]
    n = capsol.r.shape[0]
    m = capsol.z.shape[0]

    # display potential and sigma in charge-up case at solid-water interface
    fig, ax = plt.subplots(2, 2)
    i = 3
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    ax[0, 0].plot(dlf.freq, fPot[:, idx].real, '.', color='tab:blue')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel(r'$\phi$ (V)')
    ax[0, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    ax[0, 1].plot(dlf.freq, fSigma[:, idx].real, '.', color='tab:orange')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Frequency (Hz)')
    ax[0, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[0, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    # np.savetxt('pot.txt', pot[:, idx].real, fmt=' %17.8E', comments='')
    tmp = dlf.totemporal(fPot[:, idx].real)
    ax[1, 0].plot(dlf.time, tmp, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('Time (sec)')
    ax[1, 0].set_ylabel(r'$\phi$ (V)')
    ax[1, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 0].legend(['Charge-up'], loc='upper right')

    tmp = dlf.totemporal(fSigma[:, idx].real)
    ax[1, 1].plot(dlf.time, tmp, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[1, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 1].legend(['Charge-up'], loc='upper right')
    plt.tight_layout()

    # display ion concentration in charge-up case in the water
    fig, ax = plt.subplots(2, 2)
    i = 3
    j = np.where(capsol.z >= -capsol.d+1)[0][0]
    idx = i * m + j
    ax[0, 0].plot(dlf.freq, fCneg[:, idx].real, '.', color='tab:blue')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[0, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    ax[0, 1].plot(dlf.freq, fCpos[:, idx].real, '.', color='tab:orange')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Frequency (Hz)')
    ax[0, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[0, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    # np.savetxt('pot.txt', pot[:, idx].real, fmt=' %17.8E', comments='')
    tmp = dlf.totemporal(fCneg[:, idx].real)
    ax[1, 0].plot(dlf.time, tmp, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('Time (sec)')
    ax[1, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[1, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 0].legend(['Charge-up'], loc='lower left')

    tmp = dlf.totemporal(fCpos[:, idx].real)
    ax[1, 1].plot(dlf.time, tmp, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[1, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 1].legend(['Charge-up'], loc='lower left')
    plt.tight_layout()

    # display potential and sigma in charge-down case at solid-water interface
    fig, ax = plt.subplots(2, 2)
    i = 3
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    ax[0, 0].plot(dlf.freq, fPot[:, idx].real, '.', color='tab:blue')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel(r'$\phi$ (V)')
    ax[0, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    ax[0, 1].plot(dlf.freq, fSigma[:, idx].real, '.', color='tab:orange')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Frequency (Hz)')
    ax[0, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[0, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    # np.savetxt('pot.txt', pot[:, idx].real, fmt=' %17.8E', comments='')
    tmp = dlf.totemporal(fPot[:, idx].real, dcPot[idx].real)
    ax[1, 0].plot(dlf.time, tmp, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('Time (sec)')
    ax[1, 0].set_ylabel(r'$\phi$ (V)')
    ax[1, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 0].legend(['Charge-down'], loc='lower right')

    tmp = dlf.totemporal(fSigma[:, idx].real, dcSigma[idx].real)
    ax[1, 1].plot(dlf.time, tmp, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[1, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 1].legend(['Charge-down'], loc='lower right')
    plt.tight_layout()

    # display ion concentrations in the charge-down case in the water
    fig, ax = plt.subplots(2, 2)
    i = 3
    j = np.where(capsol.z >= -capsol.d+1)[0][0]
    idx = i * m + j
    ax[0, 0].plot(dlf.freq, fCneg[:, idx].real, '.', color='tab:blue')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[0, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    ax[0, 1].plot(dlf.freq, fCpos[:, idx].real, '.', color='tab:orange')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Frequency (Hz)')
    ax[0, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[0, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))

    # np.savetxt('pot.txt', pot[:, idx].real, fmt=' %17.8E', comments='')
    tmp = dlf.totemporal(fCneg[:, idx].real, dcCneg[idx].real)
    ax[1, 0].plot(dlf.time, tmp, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('Time (sec)')
    ax[1, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[1, 0].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 0].legend(['Charge-down'], loc='lower right')

    tmp = dlf.totemporal(fCpos[:, idx].real, dcCpos[idx].real)
    ax[1, 1].plot(dlf.time, tmp, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[1, 1].set_title('X = {:4.2e} m'.format(mesh.x[idx]))
    ax[1, 1].legend(['Charge-down'], loc='lower right')
    plt.tight_layout()

    # display force
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(dlf.freq, fFz[:, 0], '.')
    ax[0, 0].plot(dlf.freq, fFz[:, 1], '.')
    ax[0, 0].plot(dlf.freq, fFz[:, 2], '.')
    ax[0, 0].plot(dlf.freq, fFz[:, 3], '.')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel('Force (nN)')
    ax[0, 0].set_title('Spectral')
    ax[0, 0].legend(['bot', 'sphere', 'cone', 'disk'])

    tmp = dlf.totemporal(fFz)
    ax[0, 1].plot(dlf.time, tmp[:, 0], '.')
    ax[0, 1].plot(dlf.time, tmp[:, 1], '.')
    ax[0, 1].plot(dlf.time, tmp[:, 2], '.')
    ax[0, 1].plot(dlf.time, tmp[:, 3], '.')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Time (sec)')
    ax[0, 1].set_ylabel('Force (nN)')
    ax[0, 1].set_title('Charge-Up')
    ax[0, 1].legend(['bot', 'sphere', 'cone', 'disk'])

    # ax[1, 0].plot(dlf.time, dlf.totemporal(fFz[:, 0], dcFz[0]), '.')
    # ax[1, 0].plot(dlf.time, dlf.totemporal(fFz[:, 1], dcFz[1]), '.')
    # ax[1, 0].plot(dlf.time, dlf.totemporal(fFz[:, 2], dcFz[2]), '.')
    # ax[1, 0].plot(dlf.time, dlf.totemporal(fFz[:, 3], dcFz[3]), '.')
    # ax[1, 0].set_xscale('log')
    # ax[1, 0].set_xlabel('Time (sec)')
    # ax[1, 0].set_ylabel('Force (nN)')
    # ax[1, 0].set_title('Charge-Down')
    # ax[1, 0].legend(['bot', 'sphere', 'cone', 'disk'])

    tmp = dlf.totemporal(fFz, dcFz)
    ax[1, 1].plot(dlf.time, tmp[:, 0], '.')
    ax[1, 1].plot(dlf.time, tmp[:, 1], '.')
    ax[1, 1].plot(dlf.time, tmp[:, 2], '.')
    ax[1, 1].plot(dlf.time, tmp[:, 3], '.')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel('Force (nN)')
    ax[1, 1].set_title('Charge-Down')
    ax[1, 1].legend(['bot', 'sphere', 'cone', 'disk'])
    plt.tight_layout()

    # display force gradient
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(dlf.freq, (fFz[:, 0] - prev_fFz[:, 0])/2, '.')
    ax[0, 0].plot(dlf.freq, (fFz[:, 1] - prev_fFz[:, 1])/2, '.')
    ax[0, 0].plot(dlf.freq, (fFz[:, 2] - prev_fFz[:, 2])/2, '.')
    ax[0, 0].plot(dlf.freq, (fFz[:, 3] - prev_fFz[:, 3])/2, '.')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_ylabel('Force Grad (nN/nm)')
    ax[0, 0].set_title('Spectral')
    ax[0, 0].legend(['bot', 'sphere', 'cone', 'disk'])

    tmp = dlf.totemporal((fFz-prev_fFz)/2)
    ax[0, 1].plot(dlf.time, tmp[:, 0], '.')
    ax[0, 1].plot(dlf.time, tmp[:, 1], '.')
    ax[0, 1].plot(dlf.time, tmp[:, 2], '.')
    ax[0, 1].plot(dlf.time, tmp[:, 3], '.')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Time (sec)')
    ax[0, 1].set_ylabel('Force Grad (nN/nm)')
    ax[0, 1].set_title('Charge-Up')
    ax[0, 1].legend(['bot', 'sphere', 'cone', 'disk'])

    tmp = dlf.totemporal((fFz-prev_fFz)/2, (dcFz-prev_dcFz)/2)
    ax[1, 1].plot(dlf.time, tmp[:, 0], '.')
    ax[1, 1].plot(dlf.time, tmp[:, 1], '.')
    ax[1, 1].plot(dlf.time, tmp[:, 2], '.')
    ax[1, 1].plot(dlf.time, tmp[:, 3], '.')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel('Force Grad (nN/nm)')
    ax[1, 1].set_title('Charge-Down')
    ax[1, 1].legend(['bot', 'sphere', 'cone', 'disk'])
    plt.tight_layout()

    # display potential and surface charge density at solid-water interface
    fig, ax = plt.subplots(2, 2)
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    ax[0, 0].plot(mesh.x[idx], fPot[0, idx].real, '.', color='tab:blue')
    ax[0, 0].plot(-mesh.x[idx], fPot[0, idx].real, '.', color='tab:blue')
    ax[0, 0].set_xlim(-1e-6, 1e-6)
    ax[0, 0].set_xlabel('X (m)')
    ax[0, 0].set_ylabel(r'$\phi$ (V)')
    ax[0, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[0, 1].plot(mesh.x[idx], fSigma[0, idx].real, '.', color='tab:orange')
    ax[0, 1].plot(-mesh.x[idx], fSigma[0, idx].real, '.', color='tab:orange')
    ax[0, 1].set_xlim(-1e-6, 1e-6)
    ax[0, 1].set_xlabel('X (m)')
    ax[0, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[0, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 0].plot(mesh.x[idx], fPot[0, idx].real, '.', color='tab:blue')
    ax[1, 0].plot(-mesh.x[idx], fPot[0, idx].real, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('X (m)')
    ax[1, 0].set_ylabel(r'$\phi$ (V)')
    ax[1, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 1].plot(mesh.x[idx], fSigma[0, idx].real, '.', color='tab:orange')
    ax[1, 1].plot(-mesh.x[idx], fSigma[0, idx].real, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('X (m)')
    ax[1, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[1, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))
    plt.tight_layout()

    # display ion concentrations in the water
    fig, ax = plt.subplots(2, 2)
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d+1)[0][0]
    idx = i * m + j
    ax[0, 0].plot(mesh.x[idx], fCneg[0, idx].real, '.', color='tab:blue')
    ax[0, 0].plot(-mesh.x[idx], fCneg[0, idx].real, '.', color='tab:blue')
    ax[0, 0].set_xlim(-1e-6, 1e-6)
    ax[0, 0].set_xlabel('X (m)')
    ax[0, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[0, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[0, 1].plot(mesh.x[idx], fCpos[0, idx].real, '.', color='tab:orange')
    ax[0, 1].plot(-mesh.x[idx], fCpos[0, idx].real, '.', color='tab:orange')
    ax[0, 1].set_xlim(-1e-6, 1e-6)
    ax[0, 1].set_xlabel('X (m)')
    ax[0, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[0, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 0].plot(mesh.x[idx], fCneg[0, idx].real, '.', color='tab:blue')
    ax[1, 0].plot(-mesh.x[idx], fCneg[0, idx].real, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('X (m)')
    ax[1, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[1, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 1].plot(mesh.x[idx], fCpos[0, idx].real, '.', color='tab:orange')
    ax[1, 1].plot(-mesh.x[idx], fCpos[0, idx].real, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('X (m)')
    ax[1, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[1, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))
    plt.tight_layout()

    from sipmod.visuals import plot
    ax = plot(mesh, fPot[0, :].real)
    ax.set_xlim(-50e-6, 50e-6)
    ax.set_ylim(-50e-6, 50e-6)
    return ax


if __name__ == "__main__":
    # run()
    # validate().show()
    visualize().show()
