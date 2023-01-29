"""
This module contains uncategorized classes and methods.
"""

import os
import importlib
import time
from dataclasses import dataclass, replace
from typing import Callable, Tuple, Any, Union, Optional, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import h5py

from numpy import ndarray
from scipy import constants
from scipy.sparse import spmatrix
from scipy.sparse import coo_matrix
from h5py._hl.group import Group
from h5py._hl.dataset import Dataset


Solution = ndarray
LinearSolver = Callable[..., ndarray]


@dataclass
class PolyData:
    """Basic finite element mesh generator."""

    cpts: ndarray
    segs: ndarray
    holes: ndarray
    zones: ndarray
    scale_factor: float = 1.0

    @property
    def xverts(self):
        return self.cpts[0, self.segs[:-1, :]]

    @property
    def yverts(self):
        return self.cpts[1, self.segs[:-1, :]]

    @property
    def zverts(self):
        if self.dim() == 3:
            return self.cpts[2, self.segs[:-1, :]]
        else:
            raise NotImplementedError

    def find_xverts(self, flag: int):
        mask = self.segs[-1, :] == flag
        return self.xverts[:, mask]

    def find_yverts(self, flag: int):
        mask = self.segs[-1, :] == flag
        return self.yverts[:, mask]

    def find_zverts(self, flag: int):
        mask = self.segs[-1, :] == flag
        return self.zverts[:, mask]

    def dim(self):
        return self.cpts.shape[0] - 1

    def read(self, fname: str):
        raise NotImplementedError

    def write(self, fname: str):
        if self.dim() == 2:
            """Write poly file."""
            prefix = fname.strip('.poly')
            with open(prefix+'.poly', 'w') as f:
                #  write control points
                s = '{} 2 0 1 '\
                    '#verticies #dimensions #attributes #boundary marker'
                f.write(s.format(str(self.cpts.shape[1]))+'\n')
                for i in range(self.cpts.shape[1]):
                    s = '{:6d} {:28.18E} {:28.18E} {:6.0f}'
                    f.write(s.format(i+1, *self.cpts[:, i])+'\n')
                f.write('\n')

                # write segments
                s = '{} 1 #segments #boundary marker'
                f.write(s.format(self.segs.shape[1])+'\n')
                for i in range(self.segs.shape[1]):
                    a = self.segs[0, i] + 1
                    b = self.segs[1, i] + 1
                    flag = self.segs[2, i]
                    s = '{:6d} {:6d} {:6d} {:6d}'
                    f.write(s.format(i+1, a, b, flag)+'\n')
                f.write('\n')

                # write holes
                f.write('{} #holes'.format(str(self.holes.shape[1]))+'\n')
                for i in range(self.holes.shape[1]):
                    s = '{:6d} {:28.18E} {:28.18E}'
                    f.write(s.format(i+1, *self.holes[:, i])+'\n')
                f.write('\n')

                # write zones
                s = '{} #regional attributes and/or area constraints'
                f.write(s.format(str(self.zones.shape[1]))+'\n')
                for i in range(self.zones.shape[1]):
                    x = self.zones[0, i]
                    y = self.zones[1, i]
                    area = self.zones[2, i]
                    s = '{:6d} {:28.18E} {:28.18E} {:6d} {:12.2E}'
                    f.write(s.format(i+1, x, y, i+1, area)+'\n')
                f.write('\n')
                f.write('# triangle -pnq30Aae '+prefix+'.poly\n')
        else:
            raise NotImplementedError

    def build(self,
              fname: str,
              executable: str = 'triangle',
              flag: str = '-pnq30Aae'):
        if self.dim() == 2:
            """Build triangle mesh."""
            prefix = fname.strip('.poly')
            os.system('{} {} {}.poly'.format(executable, flag, prefix))
        else:
            raise NotImplementedError

    def draw(self, **kwargs):
        """Convenience wrapper for sipmod.visuals."""
        mod = importlib.import_module('sipmod.visuals')
        return mod.draw(self, **kwargs)


@dataclass
class CapSolData:
    # constants
    pi: float = 3.141592653589793238462643383279502884197
    e0: float = 8.854187817620e-3  # vacuum permittivity in nN/V2
    # discretization
    n: int = 500  # Number of grids: n
    m: int = 500  # Number of grids: m+
    l_js: int = 10  # Number of grids: m-
    # simulation parameters
    h0: float = 0.5  # Resolution: h0
    rho_max: float = 1e6  # Box-size: Rho_max
    z_max: float = 1e6  # Box-size: z_max
    # tip-sample separation
    d_min: float = 2.0  # Tip-sample separation: min
    d_max: float = 20.0  # Tip-sample separation: max
    idstep: int = 2  # Tip-sample separation: istep (stepsize=istep*h0)
    # tip parameters
    Rtip: float = 20.0  # Probe shape: Rtip
    theta: float = 15.0  # Probe shape: half-angle
    HCone: float = 15e3  # Probe shape: HCone
    RCant: float = 40e3  # Probe shape: RCantilever
    dCant: float = 500.0  # Probe shape: thickness_Cantilever
    # sample parameters
    reps_s: float = 5.0  # Sample: eps_r
    Hsam: float = 500e-3  # Sample: Thickness_sample
    reps_w: float = 1.0  # Water: eps_r
    Hwat: float = 1.0  # Water: Thickness_water
    # solver parameters
    method: str = 'LAPACK'  # Solving: Method{LAPACK,PARDISO,NOSOLVE}
    test: int = 0  # Test?(0=No,1=Sphere,2=lever-only)
    verbose: int = 2  # Verbosiy>0
    # other parameters
    d: Optional[float] = None  # Tip-sample separation: in-use
    # _js: Optional[int] = None
    # _r: Optional[ndarray] = None
    # _z: Optional[ndarray] = None
    # _dr: Optional[ndarray] = None
    # _zm: Optional[ndarray] = None
    # _zl: Optional[ndarray] = None
    # _hm: Optional[ndarray] = None
    # _hl: Optional[ndarray] = None
    # _probe: Optional[ndarray] = None
    # _params: Optional[ndarray] = None

    @property
    def pi_e0(self):
        return self.pi * self.e0  # to make units for unitless quantities

    @property
    def js(self) -> float:
        if self.d is None:
            raise NotImplementedError
        return self._js

    @property
    def r(self) -> ndarray:
        if self.d is None:
            raise NotImplementedError
        return self._r

    @property
    def z(self) -> ndarray:
        if self.d is None:
            raise NotImplementedError
        return self._z

    @property
    def probe(self) -> ndarray:
        if self.d is None:
            raise NotImplementedError
        i = self._probe[0, :]
        j = self._probe[1, :]
        flags = self._probe[2, :]
        x = self._r[i]
        y = self._z[j]
        return np.array([x, y, i, j, flags])

    @property
    def params(self) -> ndarray:
        if self.d is None:
            raise NotImplementedError
        aa = self._params[0, :, :]
        bb = self._params[1, :, :]
        x, y = np.meshgrid(self._r, self._z, indexing='ij')
        dx, dy = np.meshgrid(self._dr, self._dz, indexing='ij')
        return np.array([x, y, dx, dy, aa, bb]).reshape(6, -1)

    def new(self,
            d: float = None,
            d_min: float = None,
            d_max: float = None):
        """Return a copy of the capsol data given new tip-sample separation."""
        if d is None:
            d = self.d
        if d_min is None:
            d_min = self.d_min
        if d_max is None:
            d_max = self.d_max
        h0 = self.h0
        idstep = self.idstep
        if d is None:
            return replace(self, d=d, d_min=d_min, d_max=d_max)
        elif d < d_min:
            s = np.arange(0, d_min+h0, h0)
            i = np.where(s <= d)[0][-1]  # !!! less than or greater than?
            return replace(self, d=s[i], d_min=d_min, d_max=d_min)
        else:
            s = np.arange(d_min, d_max+idstep*h0, idstep*h0)
            i = np.where(s <= d)[0][-1]  # !!! less than or greater than?
            return replace(self, d=s[i], d_min=s[i], d_max=s[i])

    def generate_poly(self) -> PolyData:
        if self.d is None:
            raise NotImplementedError
        return type(self).build_polydata(self._r,
                                         self._z,
                                         self.probe,
                                         self.d,
                                         self.Hwat)

    def post_processing(self,
                        u: ndarray,
                        path: str = None) -> Tuple[ndarray,
                                                   ndarray,
                                                   ndarray]:
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        flags = self._probe[2, :self._probe.shape[1]//2]
        fields = self.nodal_fields(u, path=path)
        Er = fields[4, :]
        Ez = fields[5, :]
        fbot, ftop = self.integrated_force(u, path=path)
        Fz = np.zeros(8)
        Fz[0] = fbot[1, -1]
        Fz[1] = np.sum(fbot[2, flags == 1])
        Fz[2] = np.sum(fbot[2, flags == 2])
        Fz[3] = np.sum(fbot[2, flags == 3])
        Fz[4] = np.sum(fbot[2, flags == 4])
        Fz[5] = np.sum(fbot[2, :])
        Fz[6] = np.sum(ftop[2, :])
        Fz[7] = Fz[5] + Fz[6]
        Fz[:] = Fz[:] * self.pi_e0
        return Er, Ez, Fz

    def write_inputs(self, path: str = None) -> None:
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        strings = [f'{self.n:6d}{self.m:6d}{self.l_js:6d}',
                   f'{self.h0:7.3f}{self.rho_max:12.2E}{self.z_max:12.2E}',
                   f'{self.d_min:10.3f}{self.d_max:10.3f}{self.idstep:4d}',
                   f'{self.Rtip:6.1f}{self.theta:6.1f}' +
                   f'{self.HCone:12.2E}{self.RCant:12.2E}{self.dCant:12.2E}',
                   f'{self.reps_s:12.3E}{self.Hsam:12.3E}',
                   f'{self.method}{self.test:5d}{self.verbose:3d}']
        comments = ['# Number of grids: n, m+, m-',  # line 1
                    '# Resolution: h0, Box-size: Rho_max, Z_max    '\
                    '*** ALL LENGTHS IN NANOMETER ***',  # line 2
                    '# Tip-sample separation: min, max, '\
                    'istep (stepsize=istep*h0)',  # line 3
                    '# Probe shape: Rtip, half-angle, HCone, RCantilever, '\
                    'thickness_Cantilever',  # line 4
                    '# Sample: eps_r, Thickness_sample',  # line 5
                    '# Solving: Method{LAPACK,PARDISO,NOSOLVE}, '\
                    'Test?(0=No,1=Sphere,2=lever-only), Verbosiy>0)']  # line 6
        with open(path+'/capsol.in', 'w') as f:
            for i in range(6):
                f.write('{:<50}{}\n'.format(strings[i], comments[i]))

    def write_probe(self, suffix: str = None, path: str = None) -> None:
        if self.d is None:
            raise NotImplementedError
        if suffix is None:
            suffix = '{:6.4f}'.format(self.d/self.Rtip)
        else:
            suffix = ''
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        header = 'rho,  z,  i,  j,  code:  1=sphere, 2=cone, 3=cant, 4=edge'
        np.savetxt(path+'/ProbeGeometry.1.dat'+suffix,
                   self.probe[:, 1:-1].T,
                   fmt='%18.8E %17.8E %5d %5d %5d',
                   header=header)

    def write_params(self, suffix: str = None, path: str = None) -> None:
        if self.d is None:
            raise NotImplementedError
        if suffix is None:
            suffix = '{:6.4f}'.format(self.d/self.Rtip)
        else:
            suffix = ''
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        header = 'rho(nm)   zm(nm)   hn(nm)   hm(nm)   aa   bb'
        np.savetxt(path+'/Parameters.1.dat'+suffix,
                   self.params.T,
                   fmt=' %17.8E'*6,
                   header=header)

    def nodal_fields(self,
                     u: ndarray,
                     suffix: str = None,
                     path: str = None,
                     test: bool = False) -> ndarray:
        def row_dot(a, b):
            return np.einsum('ij,i->ij', a, b)

        def col_dot(a, b):
            return np.einsum('ij,j->ij', a, b)

        if self.d is None:
            raise NotImplementedError
        if suffix is None:
            suffix = '{:6.4f}'.format(self.d/self.Rtip)
        else:
            suffix = ''
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        r, z, hn, hm = self._r, self._z, self._dr, self._dz
        n = r.shape[0]  # equal to self.n + 1
        m = z.shape[0]  # equal to self.l_js + self.js + self.m + 1
        u = u[:n*m+1].reshape(n, m)
        out = np.zeros((6, n, m), dtype=u.dtype)
        if test:  # perform expensive output validation
            for i in range(n):
                for j in range(m):
                    if i == 0:
                        Er = -(u[i+1, j] - u[i, j]) / hn[i]
                    elif i == n-1:
                        Er = -(u[i, j] - u[i-1, j]) / hn[i-1]
                    else:
                        Er = -(u[i+1, j] - u[i-1, j]) / (hn[i] + hn[i-1])
                    if j == 0:
                        Ez = -(u[i, j+1] - u[i, j]) / hm[j+1]
                    elif j == m-1 or z[j] == 0:
                        Ez = -(u[i, j] - u[i, j-1]) / hm[j]
                    else:
                        Ez = -(u[i, j+1] - u[i, j-1]) / (hm[j] + hm[j+1])
                    E2 = abs(Ez) ** 2 + abs(Er) ** 2  # !!! complex numbers
                    x = r[i]
                    y = z[j]
                    pot = u[i, j]
                    out[:, i, j] = [x, y, pot, E2, Er, Ez]  # expensive
        else:
            Er = out[4, :, :]
            i = 0
            Er[i, :] = -(u[i+1, :] - u[i, :]) / hn[i]
            i = -1
            Er[i, :] = -(u[i, :] - u[i-1, :]) / hn[i-1]
            i = np.arange(n)[1:-1]
            Er[i, :] = row_dot(-(u[i+1, :] - u[i-1, :]), 1 / (hn[i] + hn[i-1]))

            Ez = out[5, :, :]
            j = 0
            Ez[:, j] = -(u[:, j+1] - u[:, j]) / hm[j+1]
            j = -1
            Ez[:, j] = -(u[:, j] - u[:, j-1]) / hm[j]
            j = np.arange(m)[1:-1]
            Ez[:, j] = col_dot(-(u[:, j+1] - u[:, j-1]), 1 / (hm[j] + hm[j+1]))
            j = np.where(z >= 0)[0][0]  # equal to self.l_js + self.js
            Ez[:, j] = -(u[:, j] - u[:, j-1]) / hm[j]

            E2 = out[3, :, :]
            E2[:] = np.abs(Er) ** 2 + np.abs(Ez) ** 2

            x = out[0, :, :]
            y = out[1, :, :]
            pot = out[2, :, :]
            x[:], y[:] = np.meshgrid(r, z, indexing='ij')  # fortran-like
            pot[:] = u

        header = 'rho(nm)  z(nm)  pot(V)  E2(1/nm)2  E_rho(1/nm)  E_z(1/nm)'
        np.savetxt(path+'/Fields.1.dat'+suffix,
                   out.reshape(6, -1).T,
                   fmt=' %17.8E'*6,
                   header=header)
        return out.reshape(6, -1)

    def integrated_force(self,
                         u: ndarray,
                         suffix: str = None,
                         path: str = None) -> ndarray:
        if self.d is None:
            raise NotImplementedError
        if suffix is None:
            suffix = '{:6.4f}'.format(self.d/self.Rtip)
        else:
            suffix = ''
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        r, z, hn, hm = self._r, self._z, self._dr, self._dz
        n = r.shape[0]  # equal to self.n + 1
        m = z.shape[0]  # equal to self.l_js + self.js + self.m + 1
        u = u[:n*m+1].reshape(n, m)
        ix_bot = self._probe[0, :self._probe.shape[1]//2]
        id_bot = self._probe[1, :self._probe.shape[1]//2]
        ix_top = self._probe[0, self._probe.shape[1]//2:]
        id_top = self._probe[1, self._probe.shape[1]//2:]
        out1 = np.zeros((10, id_bot.shape[0]), dtype=u.dtype)
        force = 0.0
        for k in range(id_bot.shape[0]):
            i = ix_bot[k]
            j = id_bot[i] - 1  # avoid stepwise boundary
            Er = -(u[i+1, j] - u[i, j]) / hn[i]
            Ez = -(u[i, j] - u[i, j-1]) / hm[j]
            E2 = abs(Er) ** 2 + abs(Ez) ** 2  # !!! complex numbers
            df = 0.5 * (r[i] + r[i+1]) * hn[i] * E2
            force = force + df
            x = 0.5 * (r[i] + r[i+1])
            y = z[id_bot[i]]
            u0 = u[i, j]
            u2 = u[i+1, j]
            u3 = u[i, j-1]
            out1[:, k] = [x, force, df, y, E2, Er, Ez, u0, u2, u3]  # expensive

        out2 = np.zeros((10, id_top.shape[0]), dtype=u.dtype)
        force = 0.0
        for k in range(id_top.shape[0]):
            i = ix_top[k]
            j = id_top[i] + 1  # avoid stepwise boundary
            Er = -(u[i+1, j] - u[i, j]) / hn[i]
            Ez = -(u[i, j+1] - u[i, j]) / hm[j]
            E2 = abs(Er) ** 2 + abs(Ez) ** 2  # !!! complex numbers
            df = 0.5 * (r[i] + r[i+1]) * hn[i] * E2
            force = force + df
            x = 0.5 * (r[i] + r[i+1])
            y = z[id_bot[i]]
            u0 = u[i, j]
            u2 = u[i+1, j]
            u4 = u[i, j+1]
            out2[:, k] = [x, force, df, y, E2, Er, Ez, u0, u2, u4]  # expensive

        header = 'rho, F_z, dF_z, z(rho), E2 (=>sigma2=e0^2*E2), Er, Ez, '\
                 'Ui,j , Ui+1,j , Ui,j-1'
        np.savetxt(path+'/Fz.1.dat'+suffix,
                   out1.T,
                   fmt=' %17.8E'*10,
                   header=header)
        header = 'rho, F_z, dF_z, z(rho), E2 (=>sigma2=e0^2*E2), Er, Ez, '\
                 'Ui,j , Ui+1,j , Ui,j+1'
        np.savetxt(path+'/Fz.2.dat'+suffix,
                   out2.T,
                   fmt=' %17.8E'*10,
                   header=header)
        return out1, out2

    def integrated_energy(self,
                          u: ndarray,
                          suffix: str = None,
                          path: str = None) -> ndarray:
        if self.d is None:
            raise NotImplementedError
        if suffix is None:
            suffix = '{:6.4f}'.format(self.d/self.Rtip)
        else:
            suffix = ''
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)

        n = self._r.shape[0]  # equal to self.n + 1
        m = self._z.shape[0]  # equal to self.l_js + self.js + self.m + 1
        aa = self._params[0, :, :]
        bb = self._params[1, :, :]
        u = u[:n*m+1].reshape(n, m)
        energy = 0.0
        for i in range(n-1):
            for j in range(1, m):
                energy = energy + (aa[i, j] * (u[i, j] - u[i, j-1]) ** 2 +
                                   bb[i, j] * (u[i+1, j] - u[i, j]) ** 2)
        header = '  Z/R    U/pie0RV2    C/pie0R'
        out = np.array([self.d/self.Rtip,
                        energy/self.Rtip,
                        2*energy/self.Rtip])
        if os.path.isfile(path+'/Z-U.1.dat') and suffix == '':
            with open(path+'/Z-U.1.dat', 'a') as f:
                f.write(' {:17.8E} {:17.8E} {:17.8E}'.format(*out)+'\n')
        else:
            with open(path+'/Z-U.1.dat'+suffix, 'w') as f:
                f.write('# '+header+'\n')
                f.write(' {:17.8E} {:17.8E} {:17.8E}'.format(*out)+'\n')
        return out

    def integrated_moment(self, u: ndarray, order: int = 0) -> ndarray:
        if self.d is None:
            raise NotImplementedError

        r, hn = self._r, self._dr
        n = self._r.shape[0]  # equal to self.n + 1
        m = self._z.shape[0]  # equal to self.l_js + self.js + self.m + 1
        u = u[:n*m+1].reshape(n, m)
        i = np.arange(n-1)
        j = np.where(self._z >= -self.d)[0][0]
        dm = (0.5 * (r[i] + r[i+1])) ** (order + 1) * hn[i] * u[i, j]
        return 2 * np.pi * np.sum(dm)

    @staticmethod
    def build_rho(n: int,
                  h0: float,
                  rho_max: float) -> ndarray:
        # modified from CapSol GenerateGrid
        # allocate (hn(0:n), r(0:n), hm(-l:m), zm(-l:m))
        hn = np.zeros(n+1)
        r = np.zeros(n+1)

        Nuni = 1
        for i in range(0, Nuni):
            hn[i] = h0  # hn[0] = h0
            r[i] = h0 * i  # r[0] = 0.

        r[Nuni] = h0 * Nuni  # r[1] = h0

        # find the growth factor
        for qn in np.arange(1.0+1e-4, 1.5+1e-4, 1e-4):
            # sum of geometric series
            x = h0 * (1 - qn ** (n - Nuni)) / (1 - qn)
            if x >= rho_max - r[Nuni]:
                break  # found

        hn[Nuni] = h0 * (rho_max - r[Nuni]) / x  # hn[1]
        r[Nuni+1] = np.sum(hn[0:Nuni+1])  # r[2]
        for i in range(Nuni+2, n+1):
            hn[i-1] = hn[i-2] * qn  # hn[2] to hn[n-1]
            r[i] = np.sum(hn[0:i])  # r[3] to r[n]

        hn[n] = hn[n-1] * qn  # hn[n]
        assert np.all(np.diff(r) > 0), "grid is not monotonic increasing"
        return r, hn

    @staticmethod
    def build_zm(m: int,
                 h0: float,
                 z_max: float) -> ndarray:
        # modified from CapSol GenerateGrid
        # allocate (hn(0:n), r(0:n), hm(-l:m), zm(-l:m))
        hm = np.zeros(m+1)
        zm = np.zeros(m+1)

        Nuni = 1
        hm[1:Nuni+1] = h0  # hm[1] = h0
        for j in range(1, Nuni+1):
            zm[j] = h0 * j  # zm[1] = h0

        # find the growth factor
        for qm in np.arange(1.0+1e-4, 1.5+1e-4, 1e-4):
            # sum of geometric series
            x = h0 * (1 - qm ** (m - Nuni)) / (1 - qm)
            if x >= z_max - zm[Nuni]:
                break  # found

        hm[Nuni+1] = h0 * (z_max - zm[Nuni]) / x  # hm[2]
        zm[Nuni+1] = np.sum(hm[1:Nuni+2])  # zm[2]
        for j in range(Nuni+2, m+1):
            hm[j] = hm[j-1] * qm  # hm[3] to hm[m]
            zm[j] = np.sum(hm[1:j+1])  # zm[3] to zm[m]

        assert np.all(np.diff(zm) > 0), "grid is not monotonic increasing"
        return zm[1:], hm[1:]

    @staticmethod
    def build_zl(l_js: int,
                 h0: float,
                 d_min: float,
                 Hsam: float) -> ndarray:
        # modified from CapSol GenerateGrid
        # allocate (hn(0:n), r(0:n), hm(-l:m), zm(-l:m))
        js = int(d_min/h0)
        hl = np.zeros(l_js+js+1)
        zl = np.zeros(l_js+js+1)

        hl[0:js+1] = h0  # gap fiiled with h0
        for j in range(0, js+1):
            zl[j] = h0 * j

        # hl[0] to hl[js-1] are fixed to h0 while js increases with separation
        if l_js > 0:
            # find the growth factor
            for ql in np.arange(1.0+1e-4, 2.0+1e-4, 1e-4):
                # sum of geometric series
                x = h0 * (1 - ql ** l_js) / (1 - ql)
                if x >= Hsam:
                    break  # found
            hl[js] = h0 * Hsam / x
        else:
            hl[js] = h0

        for j in range(js+1, l_js+js+1):
            hl[j] = hl[j-1] * ql
            zl[j] = np.sum(hl[0:j])

        assert np.all(np.diff(zl) > 0), "grid is not monotonic increasing"
        return -np.flipud(zl), np.flipud(hl)

    @staticmethod
    def build_probe(r: ndarray,
                    z: ndarray,
                    Rtip: float,
                    theta: float,
                    HCone: float,
                    RCant: float,
                    dCant: float) -> ndarray:
        n = r.shape[0] - 1
        theta = theta * np.pi / 180.
        Ra = Rtip * (1.0 - np.sin(theta))
        Rc = Rtip * np.cos(theta)

        id_bot = np.zeros(n, dtype=int)
        id_top = np.zeros(n, dtype=int)
        j = np.where(z >= HCone + dCant)[0][0]  # z-min index
        id_top[:] = j

        nApex, nCone, nLever, nEdge = 0, 0, 0, 0
        for i in range(n-1):
            x = r[i]
            if x < Rc:
                # Exact tip bottom is an arc given as (x,y)
                # where (x-0)^2+(y-Rtip)^2 equals to Rtip^2

                # The arc starts at (0,0) and ends at (Rc,Ra)
                # which corresponding to angle (-np.pi/2) and (-theta)

                # Approximate tip bottom is given as (r[i],z[id_bot[i]])
                # where z[id_bot[i]] is the min(z) above y(x) on the FD grid

                nApex = i
                y = Rtip - np.sqrt(Rtip ** 2 - x ** 2)  # from (0,0) to (Rc,Ra)
                j = np.where(z >= y)[0][0]  # z-min index
                id_bot[i] = j
                j = np.where(z >= HCone+dCant)[0][0]  # z-min index
                id_top[i] = j
            elif x < (HCone-Ra)*np.tan(theta)+Rc:
                # Exact cone bottom is a line given as (x,y)
                # where (x-Rc)/(y-Ra) equals to np.tan(theta)

                # The line starts at (Rc,Ra) and ends at (x,HCone)
                # where x equals to (HCone-Ra)*np.tan(theta)+Rc

                nCone = i
                y = (x - Rc) / np.tan(theta) + Ra  # from (Rc,Ra) to (x,HCone)
                j = np.where(z >= y)[0][0]  # z-min index
                id_bot[i] = j
                j = np.where(z >= HCone+dCant)[0][0]  # z-min index
                id_top[i] = j
            elif x < RCant:
                # Exact cantilever disk bottom is a line given as (x,y)
                # where y = HCone

                nLever = i
                y = HCone
                j = np.where(z >= y)[0][0]
                id_bot[i] = j  # z-min index
                j = np.where(z >= HCone+dCant)[0][0]
                id_top[i] = j  # z-min index
            elif x < RCant+dCant/2:
                # Exact cantilever disk edge is an arc given as (x,y)
                # where (x-RCant)^2+(y-HCone-dCant/2)^2 equal to (dCant/2)^2

                nEdge = i
                tmp = np.sqrt((dCant / 2) ** 2 - (x - RCant) ** 2)
                y = HCone + dCant / 2 - tmp
                j = np.where(z >= y)[0][0]  # z-min index
                id_bot[i] = j
                j = np.where(z >= y+2*tmp)[0][0]  # z-min index
                id_top[i] = j

        indices = np.hstack((np.arange(nApex+1),
                             np.arange(nApex+1, nCone+1),
                             np.arange(nCone+1, nLever+1),
                             np.arange(nLever+1, nEdge+1)))
        flags = np.hstack((np.arange(nApex+1)*0+1,
                           np.arange(nApex+1, nCone+1)*0+2,
                           np.arange(nCone+1, nLever+1)*0+3,
                           np.arange(nLever+1, nEdge+1)*0+4))
        return np.array([np.hstack((indices, np.flipud(indices))),
                         np.hstack((id_bot[indices],
                                    np.flipud(id_top[indices]))),
                         np.hstack((flags, np.flipud(flags)))])

    @staticmethod
    def build_params(r: ndarray,
                     z: ndarray,
                     hn: ndarray,
                     hm: ndarray,
                     d_min: float,
                     reps_s: float,
                     reps_w: float,
                     Hwat: float) -> ndarray:
        n = r.shape[0]  # equal to self.n + 1
        m = z.shape[0]  # equal to self.l_js + self.js + self.m + 1
        l_js = np.where(z >= -d_min)[0][0]  # equal to self.l_js
        l_jw = np.where(z >= -d_min+Hwat)[0][0]
        aa = np.zeros((n, m))
        bb = np.zeros((n, m))
        for i in range(n-1):
            for j in range(l_jw+1, m):
                aa[i, j] = 0.5 * (r[i] + r[i+1]) * hn[i] / hm[j]
                bb[i, j] = 0.5 * (r[i] + r[i+1]) / hn[i] * hm[j]
            for j in range(l_js+1, l_jw+1):
                aa[i, j] = 0.5 * (r[i] + r[i+1]) * hn[i] / hm[j] * reps_w
                bb[i, j] = 0.5 * (r[i] + r[i+1]) / hn[i] * hm[j] * reps_w
            for j in range(1, l_js+1):
                aa[i, j] = 0.5 * (r[i] + r[i+1]) * hn[i] / hm[j] * reps_s
                bb[i, j] = 0.5 * (r[i] + r[i+1]) / hn[i] * hm[j] * reps_s
        return np.array([aa, bb])

    @staticmethod
    def build_polydata(r: ndarray,
                       z: ndarray,
                       probe: ndarray,
                       d_min: float,
                       Hwat: float) -> PolyData:
        def control_points(r, z):
            x, y = np.meshgrid(r, z, indexing='ij')   # fortran-like
            flags = np.zeros((r.shape[0], z.shape[0]))
            return np.array([x, y, flags]).reshape(3, -1)

        def segments(r, z, d_min, Hwat):
            n = r.shape[0]  # self.n + 1
            m = z.shape[0]  # self.l_js + self.js + self.m + 1

            # solid-water interface
            sw = np.zeros((3, n-1), dtype=int)
            i = np.arange(n-1)
            j = np.where(z >= -d_min)[0][0]
            sw[0, :] = i * m + j
            sw[1, :] = (i + 1) * m + j
            sw[2, :] = 1

            # air-water interface
            aw = np.zeros((3, n-1), dtype=int)
            i = np.arange(n-1)
            j = np.where(z >= -d_min+Hwat)[0][0]
            aw[0, :] = i * m + j
            aw[1, :] = (i + 1) * m + j
            aw[2, :] = 2

            # equipotential surface
            ep = np.zeros((3, probe.shape[1]-1), dtype=int)
            i = probe[2, :]
            j = probe[3, :]
            ep[0, :] = i[:-1] * m + j[:-1]
            ep[1, :] = i[1:] * m + j[1:]
            ep[2, :] = 3

            # axis of symmetry (left boundary)
            aos = np.zeros((3, m-1), dtype=int)
            i = 0
            j = np.arange(m-1)
            aos[0, :] = i * m + j
            aos[1, :] = i * m + j + 1
            aos[2, :] = 4

            # right boundary
            right = np.zeros((3, m-1), dtype=int)
            i = n - 1
            j = np.arange(m-1)
            right[0, :] = i * m + j
            right[1, :] = i * m + j + 1
            right[2, :] = 14

            # bottom boundary
            bottom = np.zeros((3, n-1), dtype=int)
            i = np.arange(n-1)
            j = 0
            bottom[0, :] = i * m + j
            bottom[1, :] = (i + 1) * m + j
            bottom[2, :] = 12

            # top boundary
            top = np.zeros((3, n-1), dtype=int)
            i = np.arange(n-1)
            j = m - 1
            top[0, :] = i * m + j
            top[1, :] = (i + 1) * m + j
            top[2, :] = 11

            return np.hstack((sw, aw, ep, aos, right, bottom, top))

        cpts = control_points(r, z)
        segs = segments(r, z, d_min, Hwat)
        x = [0.5 * (probe[0, 1] + probe[0, -2])]
        y = [0.5 * (probe[1, 1] + probe[1, -2])]
        holes = np.array([x, y])
        x = [r[1], r[1], r[1]]
        y = [z[1], -d_min+0.5*Hwat, z[-1]]
        area = [r.max()*z.max()]*3
        zones = np.array([x, y, area])
        return PolyData(cpts, segs, holes, zones, scale_factor=1.0)

    def __post_init__(self):
        if self.d is not None:
            assert self.d >= 0, "negative tip-sample separation found"
            assert self.d_max >= self.d_min, "d_max is less than d_min"

            self._js = int(self.d_min/self.h0)  # !!! grid
            self._r, self._dr = type(self).build_rho(self.n,
                                                     self.h0,
                                                     self.rho_max)
            self._zm, self._hm = type(self).build_zm(self.m,
                                                     self.h0,
                                                     self.z_max)
            self._zl, self._hl = type(self).build_zl(self.l_js,
                                                     self.h0,
                                                     self.d_min,  # !!! grid
                                                     self.Hsam)
            self._z = np.hstack((self._zl, self._zm))
            self._dz = np.hstack((self._hl, self._hm))
            self._probe = type(self).build_probe(self._r,
                                                 self._z,
                                                 self.Rtip,
                                                 self.theta,
                                                 self.HCone,
                                                 self.RCant,
                                                 self.dCant)
            self._params = type(self).build_params(self._r,
                                                   self._z,
                                                   self._dr,
                                                   self._dz,
                                                   self.d,  # !!! coefficients
                                                   self.reps_s,
                                                   self.reps_w,
                                                   self.Hwat)


@dataclass
class COOData:
    """Modified from skfem.assembly.form.coo_data.COOData."""
    indices: ndarray
    data: ndarray
    shape: Tuple[int, ...]

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, int):
            return self

        return replace(
            self,
            indices=np.hstack((self.indices, other.indices)),
            data=np.hstack((self.data, other.data)),
            shape=tuple(max(self.shape[i],
                            other.shape[i]) for i in range(len(self.shape))),
        )

    def astuple(self):
        return self.indices, self.data, self.shape

    def tocsr(self):
        """Return a sparse SciPy CSR matrix."""
        if len(self.shape) == 2:
            K = coo_matrix(
                (self.data, (self.indices[0], self.indices[1])),
                shape=self.shape)
            K.eliminate_zeros()
            return K.tocsr()

    def toarray(self) -> ndarray:
        """Return a dense numpy array."""
        if len(self.shape) == 1:
            return coo_matrix(
                (self.data, (self.indices[0], np.zeros_like(self.indices[0]))),
                shape=self.shape + (1,),
            ).toarray().T[0]
        elif len(self.shape) == 2:
            return self.tocsr().toarray()

    def todefault(self) -> Any:
        """Return the default data type.

        Scalar for 0-tensor, numpy array for 1-tensor, scipy csr matrix for
        2-tensor, self otherwise.

        """
        if len(self.shape) == 0:
            return np.sum(self.data, axis=0)
        elif len(self.shape) == 1:
            return self.toarray()
        elif len(self.shape) == 2:
            return self.tocsr()
        return self


@dataclass
class Physics:
    T: float  # ambient temperature [K]
    reps_w: float  # relative permittivity in electrolyte [SI]
    reps_i: float  # relative permittivity in solid [SI]
    c: ndarray  # intrinsic ion concentrations [mol/m^3]
    z: ndarray  # ion valences
    m_w: ndarray  # ion mobility in electrolyte [m^2/(Vs)]
    s: ndarray  # surface charge density magnitudes [C]
    p: ndarray  # surface charge density signs
    m_s: ndarray  # ion mobility at SW interface [m^2/(Vs)]
    R: float = 0.0  # radius of spherical particle
    u_0: Union[float, complex] = 0.0  # voltage on equipotential surface
    e_0: ndarray = np.zeros(3)  # electric field in x/y/z directions
    perfect_conductor: bool = False  # True if solid is PEC otherwise False
    steady_state: bool = False  # True if steady state otherwise False
    eliminate_conc: bool = False  # True if eliminate unknown concentrations
    eliminate_sigma: bool = False  # True if eliminate unknown sigma
    water_only: bool = False  # True if water is the only simulation domain

    @property
    def nions(self):
        return self.c.shape[0]

    @property
    def kT(self):
        return constants.k * self.T

    @property
    def eps_i(self):
        return constants.epsilon_0 * self.reps_i

    @property
    def eps_w(self):
        return constants.epsilon_0 * self.reps_w

    @property
    def eps_a(self):  # will deprecate eps_a
        return constants.epsilon_0

    @property
    def eps_0(self):
        return constants.epsilon_0

    @property
    def C(self):
        return constants.N_A * self.c

    @property
    def Q(self):
        return constants.e * self.z

    @property
    def D_s(self):
        return self.m_s * self.kT / constants.e

    @property
    def D_w(self):
        return self.m_w * self.kT / constants.e

    @property
    def kappa(self):
        return np.sqrt(2 * self.Q**2 * self.C / (self.eps_w * self.kT))

    @property
    def debye_length(self):
        return 1 / self.kappa

    @property
    def sigma_s(self):  # Stern layer surface charge density
        return self.s[0] * self.p[0]

    @property
    def sigma_i(self):  # intrinsic surface charge density
        return self.s[1] * self.p[1]

    @property
    def sigma_w(self):  # Diffuse layer surface charge density
        return - (self.sigma_i + self.sigma_s)


def build_pc_ilu(A: spmatrix,
                 drop_tol: Optional[float] = 1e-4,
                 fill_factor: Optional[float] = 20) -> spl.LinearOperator:
    """Incomplete LU preconditioner."""
    P = spl.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    M = spl.LinearOperator(A.shape, matvec=P.solve)
    return M


def build_pc_diag(A: spmatrix) -> spmatrix:
    """Diagonal preconditioner."""
    return sp.spdiags(1. / A.diagonal(), 0, A.shape[0], A.shape[0])


def build_pc_eye(A: spmatrix) -> spmatrix:
    """Eye preconditioner."""
    return sp.eye(A.shape[0], A.shape[1], 0)


def solve(A: spmatrix,
          b: ndarray,
          x: Optional[ndarray] = None,
          I: Optional[ndarray] = None,  # noqa
          krylov: Optional[LinearSolver] = None,
          pc: Optional[str] = 'diag',
          **kwargs) -> Solution:
    print("Solving linear system, shape={}.".format(A.shape))
    start = time.time()
    if krylov is None:
        if pc == 'diag':
            M = build_pc_diag(A)
            y = spl.spsolve(M.dot(A), M.dot(b), **kwargs)
        elif pc == 'ilu':
            M = build_pc_ilu(A)
            y = spl.spsolve(M.dot(A), M.dot(b), **kwargs)
        else:
            y = spl.spsolve(A, b, **kwargs)
    else:
        if pc == 'diag':
            M = build_pc_diag(A)
        elif pc == 'ilu':
            M = build_pc_ilu(A)
        else:
            M = build_pc_eye(A)
        y, exit_code = krylov(A, b, M=M)
        if exit_code > 0:
            print("Iterative solver did not converge.")
        else:
            print(f"{krylov.__name__} with {pc} preconditioner converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
    if x is not None and I is not None:
        z = x.copy()
        z[I] = y
    else:
        z = y
    elapsed = time.time() - start
    print("Solving done in {} seconds.".format(elapsed))
    return z


def zero_rows(M: spmatrix, rows: ndarray) -> spmatrix:
    d = np.ones(M.shape[0])
    d[rows] = 0
    diag = sp.diags(d)
    return diag.dot(M)


def zero_cols(M: spmatrix, cols: ndarray) -> spmatrix:
    d = np.ones(M.shape[1])
    d[cols] = 0
    diag = sp.diags(d)
    return M.dot(diag)


def map_global_to_local(local2global: ndarray,
                        glob_conns: ndarray) -> ndarray:
    """Map global nodal indices to local nodal indices."""
    nnodes = np.maximum(np.max(glob_conns.ravel()), np.max(local2global)) + 1
    global2local = np.zeros(nnodes, dtype=np.int32) - 1
    global2local[local2global] = np.arange(local2global.shape[0])
    loc_conns = global2local[glob_conns]
    return loc_conns


def _condense_system(A: spmatrix,
                     b: ndarray,
                     s: ndarray,
                     D: ndarray) -> Tuple[spmatrix,
                                          ndarray,
                                          ndarray,
                                          ndarray]:
    from scipy.sparse import csr_matrix
    print("Implementing Dirichlet boundary condition.")
    start = time.time()
    mask = np.ones(A.shape[0], dtype=bool)
    mask[D] = False
    I = np.where(mask)[0]  # noqa

    x = np.zeros(A.shape[0], dtype=A.dtype)
    x[D] = s
    bout = b[I] - A.dot(x)[I]

    B = zero_cols(zero_rows(A, D), D)
    B.eliminate_zeros()
    C = B.tocoo()
    row = map_global_to_local(I, C.row)
    col = map_global_to_local(I, C.col)
    data = C.data
    Aout = csr_matrix((data, (row, col)), shape=(I.shape[0], I.shape[0]))
    elapsed = time.time() - start
    print("Implementing finshed in {} seconds.".format(elapsed))
    return Aout, bout, x, I


def condense(A, b, s, D):  # alias to _condense_system
    return _condense_system(A, b, s, D)


def _enforce_system(A: spmatrix,
                    b: ndarray,
                    s: ndarray,
                    D: ndarray,
                    diag: Union[int, ndarray] = 1.0) -> Tuple[spmatrix,
                                                              ndarray]:
    print("Implementing Dirichlet boundary condition.")
    start = time.time()
    x = np.zeros(A.shape[0], dtype=A.dtype)
    x[D] = s

    bout = b - A.dot(x)
    bout[D] = s * diag

    M = coo_matrix((np.ones_like(D)*diag, (D, D)), shape=A.shape)
    Aout = zero_cols(zero_rows(A, D), D) + M
    elapsed = time.time() - start
    print("Implementing finshed in {} seconds.".format(elapsed))
    return Aout, bout


def enforce(A, b, s, D, diag=1.0):  # alias to  _enforce_system
    return _enforce_system(A, b, s, D, diag)


def set_first_kind_bc(A: spmatrix,
                      b: ndarray,
                      s: ndarray,
                      D: ndarray,
                      condense: bool = False) -> Tuple[spmatrix, ndarray]:
    if condense:
        return _condense_system(A, b, s, D)
    else:
        return _enforce_system(A, b, s, D)


def save_dict(hf_prefix: str, data: Dict):
    hf_prefix = hf_prefix.strip('.h5')
    with h5py.File(hf_prefix+'.h5', 'w') as hf:
        print('Writing {}.h5'.format(hf_prefix))
        for ky, val in data.items():
            try:
                if val is None:
                    hf.create_dataset(ky, data=h5py.Empty('f8'))
                elif isinstance(val, str):
                    hf.create_dataset(ky, data=np.string_(val))
                elif isinstance(val, ndarray):
                    hf.create_dataset(ky, data=val)
                elif isinstance(val, int):
                    hf.create_dataset(ky, data=val)
                elif isinstance(val, float):
                    hf.create_dataset(ky, data=val)
                elif isinstance(val, complex):
                    hf.create_dataset(ky, data=val)
                elif isinstance(val, bool):
                    hf.create_dataset(ky, data=val)
                else:
                    raise NotImplementedError
            except NotImplementedError:
                print('Not supported: {0}'.format(type(val)))
            except TypeError:
                print('Error saving: {0}'.format(type(val)))


def load_dict(hf_prefix: str, key: Optional[str] = None) -> Any:
    hf_prefix = hf_prefix.strip('.h5')
    with h5py.File(hf_prefix+'.h5', 'r') as hf:
        print('Reading {}.h5'.format(hf_prefix))
        if key is not None:
            data = None
            if key in hf.keys():
                try:
                    value = hf[key]
                    if isinstance(value, Dataset):
                        if value.shape is None:
                            data = None
                        elif len(value.shape) == 0:
                            if np.issubdtype(value.dtype, np.string_):
                                data = np.string_(value).astype(str)
                            else:
                                data = np.array(value, dtype=value.dtype)
                        else:
                            data = value[:]
                    elif isinstance(value, Group):
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                except NotImplementedError:
                    print('Not supported: {0}'.format(type(value)))
        else:
            data = {}
            for ky, val in hf.items():
                try:
                    if isinstance(val, Dataset):
                        if val.shape is None:
                            data[ky] = None
                        elif len(val.shape) == 0:
                            if np.issubdtype(val.dtype, np.string_):
                                data[ky] = np.string_(val).astype(str)
                            else:
                                data[ky] = np.array(val, dtype=val.dtype)
                        else:
                            data[ky] = val[:]
                    elif isinstance(val, Group):
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                except NotImplementedError:
                    print('Not supported: {0}'.format(type(val)))
    return data


def amend_dict(hf_prefix: str, data: Dict, replace_row: int = None):
    hf_prefix = hf_prefix.strip('.h5')
    if os.path.isfile(hf_prefix+'.h5'):
        with h5py.File(hf_prefix+'.h5', 'a') as hf:
            print('Overwriting {}.h5'.format(hf_prefix))
            for ky, val in data.items():
                try:
                    if isinstance(val, ndarray):
                        if ky in hf:
                            if replace_row is None:
                                try:
                                    hf[ky][...] = val
                                except TypeError:
                                    del hf[ky]
                                    hf.create_dataset(ky, data=val)
                            else:
                                try:
                                    hf[ky][replace_row] = val
                                except TypeError:
                                    print('Error replacing: {0}'.format(
                                        type(val)))
                        else:
                            hf.create_dataset(ky, data=val)
                    else:
                        raise NotImplementedError
                except NotImplementedError:
                    print('Not supported: {0}'.format(type(val)))
                except TypeError:
                    print('Error saving: {0}'.format(type(val)))
    else:
        save_dict(hf_prefix, data)
