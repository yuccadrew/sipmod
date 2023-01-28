"""
Solve 3D Poisson Nernst Planck equations for AFM feldspar model with 100 nm
tip-sample separation.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
import scipy.sparse.linalg as spl  # noqa
from sipmod import MeshTri, Physics, FourierDLF, condense, solve  # noqa
from sipmod import PoissonNernstPlanck as PNP  # noqa
from docs.afm_feldspar.ex21 import capsol, mesh_prefix  # noqa


mesh = MeshTri.read(mesh_prefix+'.1',
                    axis_of_symmetry='Y',
                    scale_factor=1e-9)

# https://gpg.geosci.xyz/content/physical_properties/tables/permittivity_minerals.html
phys = Physics(T=293.0,
               reps_w=80.0,  # relative permittivity in water
               reps_i=6.0,  # relative permittivity in solid
               c=np.array([0, 0]),  # no ions
               z=np.array([-2, 2]),  # for calcite model
               m_w=np.array([4e-14, 4e-14]),  # ~ diffusivity as 1e-15 m^2/s
               s=np.array([0.29, 0.29]),  # 1.8 e/nm^2
               p=np.array([1, -1]),  # !!! take care for decoupled system
               m_s=np.array([4e-14, 0]),  # lead to diffusivity as 1e-15 m^2/s
               u_0=10.0,  # probe voltage
               e_0=np.array([0, 0, 0]))  # no external electric field

dlf = FourierDLF(time=np.logspace(-6, 8, 301))


def run():
    from tqdm import tqdm
    from sipmod import CellBasisTri, FacetBasisTri
    from sipmod.utils import amend_dict
    path = os.path.dirname(mesh_prefix)
    basis = {'water': CellBasisTri(mesh, 'water'),
             'solid': CellBasisTri(mesh, 'solid'),
             'air': CellBasisTri(mesh, 'air'),
             'stern': FacetBasisTri(mesh, 'stern')}
    dof = 4  # must be 4 in this case
    # scale_factor = 1 / np.array([1e-18, 1e-18, phys.eps_0, 1e-18])
    form_args = {'dtype': np.complex128,
                 'aos': 'Y',
                 'dof': dof,
                 'prev': np.zeros(mesh.nnodes*dof),
                 'freq': 0.0,
                 'scale_factor': None,
                 'basis': basis}
    A, b = PNP.assemble(mesh, phys, **form_args)
    s, D = PNP.dirichlet(mesh, phys, **form_args)
    x = solve(*condense(A, b, s, D))

    # postprocess and write to the file
    pot = x.reshape(-1, 4)[:, 2]
    probe = capsol.probe
    flags = probe[-1, :probe.shape[1]//2]
    fields = capsol.nodal_fields(pot.real, path=path)
    dcEr = fields[4, :]
    dcEz = fields[5, :]
    fbot, ftop = capsol.integrated_force(pot.real, path=path)
    dcFz = np.zeros(8)
    dcFz[0] = fbot[1, -1]
    dcFz[1] = np.sum(fbot[2, flags == 1])
    dcFz[2] = np.sum(fbot[2, flags == 2])
    dcFz[3] = np.sum(fbot[2, flags == 3])
    dcFz[4] = np.sum(fbot[2, flags == 4])
    dcFz[5] = np.sum(fbot[2, :])
    dcFz[6] = np.sum(ftop[2, :])
    dcFz[7] = dcFz[5] + dcFz[6]
    dcFz[:] = dcFz[:] * capsol.pi_e0
    amend_dict(path+'/ex22t0.h5', data={'x': mesh.x,
                                        'y': mesh.y,
                                        'freq': dlf.freq,
                                        'time': dlf.time,
                                        'dcEM': x,
                                        'dcEr': dcEr,
                                        'dcEz': dcEz,
                                        'dcFz': dcFz})

    x = np.zeros((dlf.freq.shape[0], mesh.nnodes*dof), dtype=np.complex128)
    fEr = np.zeros((dlf.freq.shape[0], mesh.nnodes))
    fEz = np.zeros((dlf.freq.shape[0], mesh.nnodes))
    fFz = np.zeros((dlf.freq.shape[0], 8))
    for i in tqdm(range(dlf.freq.shape[0])):
        print('Calculating frequency {:4.2e} Hz'.format(dlf.freq[i]))
        form_args['freq'] = dlf.freq[i]
        A, b = PNP.assemble(mesh, phys, **form_args)
        s, D = PNP.dirichlet(mesh, phys, **form_args)
        x[i, :] = solve(*condense(A, b, s, D))

        # postprocess and write to the file
        pot = x[i, :].reshape(-1, 4)[:, 2]
        fields = capsol.nodal_fields(pot.real, path=path)
        fEr[i, :] = fields[4, :]
        fEz[i, :] = fields[5, :]
        fbot, ftop = capsol.integrated_force(pot.real, path=path)
        fFz[i, 0] = fbot[1, -1]
        fFz[i, 1] = np.sum(fbot[2, flags == 1])
        fFz[i, 2] = np.sum(fbot[2, flags == 2])
        fFz[i, 3] = np.sum(fbot[2, flags == 3])
        fFz[i, 4] = np.sum(fbot[2, flags == 4])
        fFz[i, 5] = np.sum(fbot[2, :])
        fFz[i, 6] = np.sum(ftop[2, :])
        fFz[i, 7] = fFz[i, 5] + fFz[i, 6]
        fFz[i, :] = fFz[i, :] * capsol.pi_e0
        amend_dict(path+'/ex22t0.h5', data={'fEM': x,
                                            'fEr': fEr,
                                            'fEz': fEz,
                                            'fFz': fFz})


def validate():
    import matplotlib.pyplot as plt
    from sipmod.utils import load_dict
    path = os.path.dirname(mesh_prefix)
    # u = load_dict(path+'/ex22t0.h5', 'dcEM')
    u = load_dict(path+'/ex22t0.h5', 'fEM')[0, :]
    pot = u.reshape(-1, 4)[:, 2]
    sigma = u.reshape(-1, 4)[:, 3]
    cneg = u.reshape(-1, 4)[:, 0]
    cpos = u.reshape(-1, 4)[:, 1]
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

    # display ion concentrations in the water
    fig, ax = plt.subplots(2, 2)
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d+1)[0][0]
    idx = i * m + j
    ax[0, 0].plot(mesh.x[idx], cneg[idx].real, '.', color='tab:blue')
    ax[0, 0].plot(-mesh.x[idx], cneg[idx].real, '.', color='tab:blue')
    ax[0, 0].set_xlim(-1e-6, 1e-6)
    ax[0, 0].set_xlabel('X (m)')
    ax[0, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[0, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[0, 1].plot(mesh.x[idx], cpos[idx].real, '.', color='tab:orange')
    ax[0, 1].plot(-mesh.x[idx], cpos[idx].real, '.', color='tab:orange')
    ax[0, 1].set_xlim(-1e-6, 1e-6)
    ax[0, 1].set_xlabel('X (m)')
    ax[0, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[0, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 0].plot(mesh.x[idx], cneg[idx].real, '.', color='tab:blue')
    ax[1, 0].plot(-mesh.x[idx], cneg[idx].real, '.', color='tab:blue')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('X (m)')
    ax[1, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[1, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    ax[1, 1].plot(mesh.x[idx], cpos[idx].real, '.', color='tab:orange')
    ax[1, 1].plot(-mesh.x[idx], cpos[idx].real, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('X (m)')
    ax[1, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
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
    path = os.path.dirname(mesh_prefix)
    dcEM = load_dict(path+'/ex22t0.h5', 'dcEM')
    dcFz = load_dict(path+'/ex22t0.h5', 'dcFz')
    fEM = load_dict(path+'/ex22t0.h5', 'fEM')
    fFz = load_dict(path+'/ex22t0.h5', 'fFz')
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

    # display potential/sigma/ions vs distance for multiple time points
    fig, ax = plt.subplots(2, 2)
    time = np.logspace(-3, 1, 5)
    labels = ['t = {:.0e} s'.format(time[i]) for i in range(time.shape[0])]
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j

    tPot = dlf.totemporal(fPot[:, idx].real)
    tmp = dlf.interpolate(dlf.time, tPot, time, axis=0)
    ax[0, 0].plot(mesh.x[idx], tmp.T, '.')
    ax[0, 0].legend(labels, loc='best')
    ax[0, 0].set_prop_cycle(None)
    ax[0, 0].plot(-mesh.x[idx], tmp.T, '.')
    ax[0, 0].set_xlabel('X (m)')
    ax[0, 0].set_ylabel(r'$\phi$ (V)')
    ax[0, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    tSigma = dlf.totemporal(fSigma[:, idx].real)
    tmp = dlf.interpolate(dlf.time, tSigma, time, axis=0)
    ax[0, 1].plot(mesh.x[idx], tmp.T, '.')
    ax[0, 1].set_prop_cycle(None)
    ax[0, 1].plot(-mesh.x[idx], tmp.T, '.')
    ax[0, 1].set_xlim(-1e-6, 1e-6)
    ax[0, 1].set_xlabel('X (m)')
    ax[0, 1].set_ylabel(r'$\delta \sigma$ (C/m^2)')
    ax[0, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d+1)[0][0]
    idx = i * m + j
    tCneg = dlf.totemporal(fCneg[:, idx].real)
    tmp = dlf.interpolate(dlf.time, tCneg, time, axis=0)
    ax[1, 0].plot(mesh.x[idx], tmp.T, '.')
    ax[1, 0].set_prop_cycle(None)
    ax[1, 0].plot(-mesh.x[idx], tmp.T, '.')
    ax[1, 0].set_xlim(-1e-6, 1e-6)
    ax[1, 0].set_xlabel('X (m)')
    ax[1, 0].set_ylabel(r'$\delta C_-$ (mol/m^3)')
    ax[1, 0].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))

    tCpos = dlf.totemporal(fCpos[:, idx].real)
    tmp = dlf.interpolate(dlf.time, tCpos, time, axis=0)
    ax[1, 1].plot(mesh.x[idx], tmp.T, '.')
    ax[1, 1].set_prop_cycle(None)
    ax[1, 1].plot(-mesh.x[idx], tmp.T, '.')
    ax[1, 1].set_xlim(-1e-6, 1e-6)
    ax[1, 1].set_xlabel('X (m)')
    ax[1, 1].set_ylabel(r'$\delta C_+$ (mol/m^3)')
    ax[1, 1].set_title('Profile Y = {:4.2e} m'.format(mesh.y[idx][0]))
    plt.tight_layout()

    from sipmod.visuals import plot
    ax = plot(mesh, fPot[0, :].real)
    ax.set_xlim(-50e-6, 50e-6)
    ax.set_ylim(-50e-6, 50e-6)
    return ax


def plot_peaks():
    import matplotlib.pyplot as plt
    # from scipy.optimize import curve_fit
    from sipmod.utils import load_dict

    path = os.path.dirname(mesh_prefix)
    dcEM = load_dict(path+'/ex22t0.h5', 'dcEM')
    fEM = load_dict(path+'/ex22t0.h5', 'fEM')
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

    # display moment of charge
    order = 2
    dcMoment = capsol.integrated_moment(dcSigma.real, order)
    fMoment = np.zeros((dlf.freq.shape[0]))
    for i in range(dlf.freq.shape[0]):
        fMoment[i] = capsol.integrated_moment(fSigma[i, :].real, order)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(dlf.freq, fMoment, '.')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Frequency (Hz)')
    ax[0, 0].set_title('Spectral')

    tmp = dlf.totemporal(fMoment.real)
    ax[0, 1].plot(dlf.time, tmp, '.', color='tab:orange')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Time (sec)')
    ax[0, 1].set_title('Charge-Up')

    tmp = dlf.totemporal(fMoment.real, dcMoment.real)
    ax[1, 1].plot(dlf.time, tmp, '.', color='tab:green')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_title('Charge-Down')
    plt.tight_layout()

    # finding widths
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    tmp = dlf.totemporal(fSigma[:, idx].real, dcSigma[idx].real)  # charge-down
    heights = np.abs(tmp[:, 0])
    widths = np.zeros(dlf.time.shape[0])
    fig, ax = plt.subplots()
    for k in range(0, dlf.time.shape[0], 20):
        xdata = np.linspace(0, 300, 601)  # nanometer
        ydata = dlf.interpolate(capsol.r[3:], np.abs(tmp[k, :]), xdata)
        ydata = ydata / ydata[0]
        widths[k] = xdata[np.where(ydata >= ydata[0]/2)[0][-1]]
        ax.plot(xdata, ydata, '-')
        ax.plot(xdata[np.where(ydata >= ydata[0]/2)[0][-1]],
                ydata[np.where(ydata >= ydata[0]/2)[0][-1]],
                'o', color='orange')
        if abs(tmp[k, 0]) < 0.01 * abs(tmp[0, 0]):
            break

    # display charge-up sigma and ion concentrations in the water
    fig, ax = plt.subplots(2, 2)
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    tmp = dlf.totemporal(fSigma[:, idx].real)  # charge-up
    heights = np.abs(tmp[:, 0])
    widths = np.zeros(dlf.time.shape[0])
    for k in range(dlf.time.shape[0]):
        # xdata = np.linspace(0, 300, 601)  # nanometer
        ydata = dlf.interpolate(capsol.r[3:], np.abs(tmp[k, :]), xdata)
        ydata = ydata / ydata[0]
        widths[k] = xdata[np.where(ydata >= ydata[0]/2)[0][-1]]
        if abs(tmp[k, 0]) < 0.01 * abs(tmp[0, 0]):
            break
    ax[0, 0].plot(dlf.time, heights, '.', color='tab:green')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Time (sec)')
    ax[0, 0].set_ylabel('(C/m^2)')
    ax[0, 0].set_title('Peak Height')
    ax[0, 0].legend([r'$\delta \sigma$'], loc='lower left')

    ax[0, 1].plot(dlf.time[:k+1], widths[:k+1]*1e-9, '.', color='tab:green')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Time (sec)')
    ax[0, 1].set_ylabel('(m)')
    ax[0, 1].set_title('Peak Width')
    ax[0, 1].legend([r'$\delta \sigma$'], loc='lower left')
    plt.tight_layout()

    # display charge-down sigma and ion concentrations in the water
    fig, ax = plt.subplots(2, 2)
    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d)[0][0]
    idx = i * m + j
    tmp = dlf.totemporal(fSigma[:, idx].real, dcSigma[idx].real)  # charge-down
    heights = np.abs(tmp[:, 0])
    widths = np.zeros(dlf.time.shape[0])
    for k in range(dlf.time.shape[0]):
        xdata = np.linspace(0, 300, 601)  # nanometer
        ydata = dlf.interpolate(capsol.r[3:], np.abs(tmp[k, :]), xdata)
        ydata = ydata / ydata[0]
        widths[k] = xdata[np.where(ydata >= ydata[0]/2)[0][-1]]
        if abs(tmp[k, 0]) < 0.01 * abs(tmp[0, 0]):
            break
    ax[0, 0].plot(dlf.time, heights, '.', color='tab:green')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Time (sec)')
    ax[0, 0].set_ylabel('(C/m^2)')
    ax[0, 0].set_title('Peak Height')
    ax[0, 0].legend([r'$\delta \sigma$'], loc='lower left')

    ax[0, 1].plot(dlf.time[:k+1], widths[:k+1]*1e-9, '.', color='tab:green')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_xlabel('Time (sec)')
    ax[0, 1].set_ylabel('(m)')
    ax[0, 1].set_title('Peak Width')
    ax[0, 1].legend([r'$\delta \sigma$'], loc='lower left')

    i = np.arange(3, n)
    j = np.where(capsol.z >= -capsol.d+1)[0][0]
    idx = i * m + j
    tmp = dlf.totemporal(fCneg[:, idx].real, dcCneg[idx].real)  # charge-down
    heights = np.abs(tmp[:, 0])
    widths = np.zeros(dlf.time.shape[0])
    for k in range(dlf.time.shape[0]):
        xdata = np.linspace(0, 300, 601)  # nanometer
        ydata = dlf.interpolate(capsol.r[3:], np.abs(tmp[k, :]), xdata)
        widths[k] = xdata[np.where(ydata >= ydata[0]/2)[0][-1]]
        if abs(tmp[k, 0]) < 0.01 * abs(tmp[0, 0]):
            break
    ax[1, 0].plot(dlf.time, heights, '.', color='tab:orange')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('Time (sec)')
    ax[1, 0].set_ylabel('(mol/m^3)')
    ax[1, 0].set_title('Peak Height')
    ax[1, 0].legend([r'$\delta C_-$'], loc='lower left')

    ax[1, 1].plot(dlf.time[:k+1], widths[:k+1]*1e-9, '.', color='tab:orange')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Time (sec)')
    ax[1, 1].set_ylabel('(m)')
    ax[1, 1].set_title('Peak Width')
    ax[1, 1].legend([r'$\delta C_-$'], loc='lower left')
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    # run()
    # validate().show()
    # visualize().show()
    plot_peaks().show()
