import importlib
import time

from typing import Callable, Optional, Union, Type, Dict, List, Tuple, Any

import numpy as np
from numpy import ndarray

from .mesh import Mesh
from .utils import Physics, COOData


def line_rotation_matrix(xverts: ndarray,
                         yverts: ndarray) -> ndarray:
    """
    Return matrix R to rotate a 2D line to direction ``x``.
    """
    A = np.zeros((3, 3))
    R = np.zeros((3, 3))
    u = np.array([xverts[1] - xverts[0], yverts[1] - yverts[0], 0])
    len_u = np.linalg.norm(u)
    u = u / len_u
    v = np.array([1, 0, 0])
    k = np.cross(u, v)
    len_k = np.linalg.norm(k)
    if not np.isclose(0, len_k):
        k = k / len_k
        cosb = np.dot(u, v)
        sinb = np.sqrt(1 - cosb ** 2)
        A[0, :] = [0, -k[2], k[1]]
        A[1, :] = [k[2], 0, -k[0]]
        A[2, :] = [-k[1], k[0], 0]
        R = np.eye(3) + A.dot(sinb) + A.dot(A.dot(1 - cosb))
    else:
        R = np.eye(3)
    return R[: -1, : -1]


def triangle_rotation_matrix(xverts: ndarray,
                             yverts: ndarray,
                             zverts: ndarray) -> ndarray:
    """
    Return matrix R to rotate a triangle normal to direction ``z``.
    """
    A = np.zeros((3, 3))
    R = np.zeros((3, 3))
    a = np.r_[xverts[1] - xverts[0],
              yverts[1] - yverts[0],
              zverts[1] - zverts[0]]
    b = np.r_[xverts[2] - xverts[1],
              yverts[2] - yverts[1],
              zverts[2] - zverts[1]]
    u = np.cross(a, b)
    len_u = np.linalg.norm(u)
    u = u / len_u
    v = np.array([0, 0, 1])
    k = np.cross(u, v)
    len_k = np.linalg.norm(k)
    if not np.isclose(0, len_k):
        k = k / len_k
        cosb = np.dot(u, v)
        sinb = np.sqrt(1 - cosb ** 2)
        A[0, :] = [0, -k[2], k[1]]
        A[1, :] = [k[2], 0, -k[0]]
        A[2, :] = [-k[1], k[0], 0]
        R = np.eye(3) + A.dot(sinb) + A.dot(A.dot(1 - cosb))
    else:
        R = np.eye(3)
    return R


class Basis:
    """Finite element basis."""

    elems: Dict[str, ndarray]
    basis: List[Tuple[Dict[str, ndarray]]]
    dx: ndarray

    @property
    def nnodes(self):
        return self._mesh.nnodes

    @property
    def nelems(self):
        return self.elems['lconns'].shape[1]

    @property
    def nverts(self):
        return self.elems['lconns'].shape[0]

    @property
    def lconns(self):
        return self.elems['lconns']

    @property
    def gconns(self):
        return self.elems['gconns']

    @property
    def xverts(self):
        return self.elems['xverts']

    @property
    def yverts(self):
        return self.elems['yverts']

    @property
    def zverts(self):
        if self.dim() == 3:
            return self.elems['zverts']
        else:
            raise NotImplementedError

    def rverts(self, aos: str = 'none'):
        if aos.lower() == 'x':
            return self.yverts
        elif aos.lower() == 'y':
            return self.xverts
        else:
            return np.ones(self.nverts)

    @property
    def x(self):
        return self.elems['xverts'].mean(axis=0)

    @property
    def y(self):
        return self.elems['yverts'].mean(axis=0)

    @property
    def z(self):
        if self.dim() == 3:
            return self.elems['zverts'].mean(axis=0)
        else:
            raise NotImplementedError

    def r(self, aos: str = 'none'):
        if aos.lower() == 'x':
            return self.y
        elif aos.lower() == 'y':
            return self.x
        else:
            return 1.

    def dim(self):
        return self._mesh.dim()

    def zeros(self, dof: int = 1, dtype=None) -> ndarray:
        """Return a zero array with same dimensions as the solution."""
        return np.zeros((self.nnodes, dof), dtype=dtype).ravel()

    def __repr__(self):
        size = sum(
            [sum([x.size]) for x in self.basis[0].values()]
        ) * 8 * len(self.basis)
        rep = ""
        rep += "<{}({}, {}) object>\n".format(type(self).__name__,
                                              type(self._mesh).__name__,
                                              type(self._name).__name__)
        rep += "  Number of elements: {}\n".format(self.nelems)
        rep += "  Number of vertices: {}\n".format(self.nverts)
        rep += "  Size: {} B".format(size)
        return rep

    def draw(self, **kwargs):
        """Convenience wrapper for sipmod.visuals."""
        mod = importlib.import_module('sipmod.visuals')
        return mod.draw(self, **kwargs)


class CellBasisTri(Basis):
    """Evaluate the basis functions for 2D triangular elements."""
    def __init__(self, mesh: Mesh, name: str = 'default'):
        print("Initializing {}({}, '{}')".format(type(self).__name__,
                                                 type(mesh).__name__,
                                                 name))
        start = time.time()
        self._mesh = mesh
        self._name = name
        self.elems = mesh.subdomain_elements(name)
        self.basis, self.area = CellBasisTri.build(**self.elems)
        elapsed = time.time() - start
        print("Initializing finished in {} seconds.".format(elapsed))

    def mesh_parameters(self, aos: str = 'none') -> Dict[str, ndarray]:
        return {
            'x': self.x,
            'y': self.y,
            'r': self.r(aos),
            'xverts': self.xverts,
            'yverts': self.yverts,
            'rverts': self.rverts(aos),
            'area': self.area
        }

    def field_parameters(self,
                         prev: Optional[ndarray] = None,
                         dof: int = 1,
                         loc_nid: bool = False) -> Dict[str, ndarray]:
        if prev is None:
            uverts = None
            u, grad = None, None
        else:
            uverts = prev[self.lconns] if loc_nid else prev[self.gconns]
            u, grad = self.interpolate(prev, dof, loc_nid).values()
        return {'uverts': uverts, 'u': u, 'grad': grad}

    def interpolate(self,
                    u: ndarray,
                    dof: int = 1,
                    loc_nid: bool = False) -> Dict[str, ndarray]:
        """Interpolate a solution from nodal vertices to elemental centers.

        Useful when a solution vector is needed in the forms, e.g., when
        evaluating functionals or when solving nonlinear problems.

        Parameters
        ----------
        u
            A solution vector.

        dof
            Number of unknowns per node.

        loc_id
            True if the solution vector is local indices based. Default is
            False.
        """
        value = np.zeros((self.nelems, dof), dtype=u.dtype)
        grad = np.zeros((2, self.nelems, dof), dtype=u.dtype)
        for k in range(dof):
            u_k = u.reshape(-1, dof)[:, k]
            uverts = u_k[self.lconns] if loc_nid else u_k[self.gconns]
            for i in range(3):
                bfun = self.basis[i]
                value[:, k] += (bfun['b'] + bfun['grad'][0, :] * self.x +
                                bfun['grad'][1, :] * self.y) * uverts[i, :]
                grad[0, :, k] += bfun['grad'][0, :] * uverts[i, :]
                grad[1, :, k] += bfun['grad'][1, :] * uverts[i, :]
        return {
            'value': value.reshape(-1,),
            'grad': grad.reshape(2, -1)
        }

    @staticmethod
    def build(xverts: ndarray,
              yverts: ndarray,
              *args, **kwargs) -> Tuple[List[Dict[str, ndarray]],
                                        ndarray]:
        """
        Evalute the basis functions for 2D triangular elements, given nodal
        coordinates in dimensions ``x`` and ``y``.

        Parameters
        ----------
        xverts
            An array of the nodal coordinates in dimension ``x``. The expected
            shape is: ``(3, Nelems)``.
        yverts
            An array of the nodal coordinates in dimension ``y``. The expected
            shape is: ``(3, Nelems)``.

        Returns
        -------
        Tuple(List[Dict[str, ndaray]], ndarray)
            Attributes of the basis functions.

        """
        nelems = xverts.shape[1]
        invA = np.zeros((3, 3, nelems))
        area = np.zeros(nelems)
        for i in range(nelems):
            A = np.ones((3, 3))
            A[1, :] = xverts[:, i]
            A[2, :] = yverts[:, i]
            invA[:, :, i] = np.linalg.inv(A)
            area[i] = np.abs(np.linalg.det(A)) / 2

        return [
            {'b': invA[0, 0, :], 'grad': invA[0, 1:, :]},
            {'b': invA[1, 0, :], 'grad': invA[1, 1:, :]},
            {'b': invA[2, 0, :], 'grad': invA[2, 1:, :]}
        ], area


class FacetBasisTri(Basis):
    """Evaluate the basis functions for 2D line elements."""
    def __init__(self, mesh: Mesh, name: str = 'default'):
        print("Initializing {}({}, '{}')".format(type(self).__name__,
                                                 type(mesh).__name__,
                                                 name))
        start = time.time()
        self._mesh = mesh
        self._name = name
        self.elems = mesh.boundary_elements(name)
        self.basis, self.dx, self.mapping = FacetBasisTri.build(**self.elems)
        elapsed = time.time() - start
        print("Initializing finished in {} seconds.".format(elapsed))

    @property
    def xnew(self):
        return self.mapping['xrots'].mean(axis=0)

    @property
    def ynew(self):
        return self.mapping['yrots'].mean(axis=0)

    def mesh_parameters(self, aos: str = 'none') -> Dict[str, ndarray]:
        return {
            'x': self.x,
            'y': self.y,
            'r': self.r(aos),
            'xverts': self.xverts,
            'yverts': self.yverts,
            'rverts': self.rverts(aos),
            'dx': self.dx
        }

    def field_parameters(self,
                         prev: Optional[ndarray] = None,
                         dof: int = 1,
                         loc_nid: bool = False) -> Dict[str, ndarray]:
        if prev is None:
            uverts = None
            u, grad = None, None
        else:
            uverts = prev[self.lconns] if loc_nid else prev[self.gconns]
            u, grad = self.interpolate(prev, dof, loc_nid).values()
        return {'uverts': uverts, 'u': u, 'grad': grad}

    def rotate(self,
               vector_field: ndarray,
               dof: int = 1) -> ndarray:
        """Return vector field in the rotated coordinates.
        Usage:
            du, dv = self.rotate(np.vstack((dx, dy)))
        """
        tmp = vector_field.reshape(2, -1, dof)
        out = np.einsum('ijk,jkl->ikl', self.mapping['R'], tmp)
        return out.reshape(vector_field.shape)

    def unrotate(self,
                 vector_field: ndarray,
                 dof: int = 1) -> ndarray:
        """Return vector field in the unrotated coordinates.
        Usage:
            dx, dy = self.unrotate(np.vstack((du, dv)))
        """
        tmp = vector_field.reshape(2, -1, dof)
        out = np.einsum('ijk,jkl->ikl', self.mapping['invR'], tmp)
        return out.reshape(vector_field.shape)

    def project(self,
                vector_field: ndarray,
                dof: int = 1) -> Dict[str, ndarray]:
        """Return vector field tangential to the edge elements.
        Usage:
            t, n = self.project(np.vstack(dudx, dudy)).values()
        """
        vector_line = np.vstack((np.diff(self.xverts, axis=0) / self.dx,
                                 np.diff(self.yverts, axis=0) / self.dx))
        tmp = vector_field.reshape(2, -1, dof)
        t = np.zeros_like(tmp)
        n = np.zeros_like(tmp)
        for k in range(dof):
            for i in range(2):
                t[i, :, k] = np.einsum('ij,ij->j',
                                       tmp[:, :, k],
                                       vector_line) * vector_line[i, :]
                n[i, :, k] = tmp[i, :, k] - t[i, :, k]
        return {
            't': t.reshape(vector_field.shape),
            'n': n.reshape(vector_field.shape)
        }

    def interpolate(self,
                    u: ndarray,
                    dof: int = 1,
                    loc_nid: bool = False,
                    return_unrotated: bool = False) -> Dict[str, ndarray]:
        """Interpolate a solution from nodal vertices to elemental centers.

        Useful when a solution vector is needed in the forms, e.g., when
        evaluating functionals or when solving nonlinear problems.

        Parameters
        ----------
        u
            A solution vector.

        dof
            Number of unknowns per node.

        loc_id
            True if the solution vector is local indices based. Default is
            False.

        return_unrotated
            Return gradient vector in the unrotated coordinates if True.
            Default is False.

        """
        value = np.zeros((self.nelems, dof), dtype=u.dtype)
        grad = np.zeros((2, self.nelems, dof), dtype=u.dtype)
        for k in range(dof):
            u_k = u.reshape(-1, dof)[:, k]
            uverts = u_k[self.lconns] if loc_nid else u_k[self.gconns]
            for i in range(2):
                bfun = self.basis[i]
                value[:, k] += (bfun['b'] +
                                bfun['grad'][0, :] * self.xnew) * uverts[i, :]
                grad[0, :, k] += bfun['grad'][0, :] * uverts[i, :]

        if return_unrotated:
            return {
                'value': value.reshape(-1,),
                'grad': self.unrotate(grad.reshape(2, -1), dof)
            }
        else:
            return {
                'value': value.reshape(-1,),
                'grad': grad.reshape(2, -1)
            }

    @staticmethod
    def build(xverts: ndarray,
              yverts: ndarray,
              *args, **kwargs) -> Tuple[List[Dict[str, ndarray]],
                                        ndarray,
                                        Dict[str, ndarray]]:
        """
        Evalute the basis functions for 2D line elements, given nodal
        coordinates in dimensions ``x`` and ``y``.

        Parameters
        ----------
        xverts
            An array of the nodal coordinates in dimension ``x``. The expected
            shape is: ``(2, Nedges)``.
        yverts
            An array of the nodal coordinates in dimension ``y``. The expected
            shape is: ``(2, Nedges)``.

        Returns
        -------
        Dict[str, ndaray]
            Dictornaty of the basis functions.

        """
        nedges = xverts.shape[1]
        xrots = np.zeros((2, nedges))
        yrots = np.zeros((2, nedges))
        R = np.zeros((2, 2, nedges))
        invR = np.zeros((2, 2, nedges))
        invA = np.zeros((2, 2, nedges))
        length = np.zeros(nedges)
        for i in range(nedges):
            tmp = line_rotation_matrix(xverts[:, i], yverts[:, i])
            xrots[0, i] = tmp[0, 0] * xverts[0, i] + tmp[0, 1] * yverts[0, i]
            yrots[0, i] = tmp[1, 0] * xverts[0, i] + tmp[1, 1] * yverts[0, i]
            xrots[1, i] = tmp[0, 0] * xverts[1, i] + tmp[0, 1] * yverts[1, i]
            yrots[1, i] = tmp[1, 0] * xverts[1, i] + tmp[1, 1] * yverts[1, i]
            R[:, :, i] = tmp
            invR[:, :, i] = np.linalg.inv(tmp)

            A = np.ones((2, 2))
            A[1, :] = xrots[:, i]
            invA[:, :, i] = np.linalg.inv(A)
            length[i] = np.abs(xrots[1, i] - xrots[0, i])

        return [
            {'b': invA[0, 0, :], 'grad': invA[0, 1:, :].reshape(1, -1)},
            {'b': invA[1, 0, :], 'grad': invA[1, 1:, :].reshape(1, -1)}
        ], length, {'xrots': xrots, 'yrots': yrots, 'R': R, 'invR': invR}


class FormExtraParams(dict):
    """Dict decorator. Passed to forms as `w`."""

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError("Attribute '{}' not found in w.".format(attr))


class Form:
    form: Optional[Callable] = None

    def __init__(self,
                 form: Optional[Callable] = None,
                 dtype: Union[Type[np.float64],
                              Type[np.complex128]] = np.float64,
                 aos: str = 'none',
                 dof: int = 1,
                 prev: Optional[ndarray] = None,
                 phys: Optional[Physics] = None,
                 freq: Optional[ndarray] = None,
                 scale_factor: Optional[ndarray] = None,
                 **params):
        self.form = form
        self.dtype = dtype
        self.aos = aos
        self.dof = dof
        self.prev = prev
        self.phys = phys
        self.freq = freq
        self.scale_factor = scale_factor
        self.params = params

    def __call__(self, *args, **kwargs):
        if self.form is None:  # decorate
            return type(self)(form=args[0],
                              dtype=self.dtype,
                              aos=self.aos,
                              dof=self.dof,
                              prev=self.prev,
                              phys=self.phys,
                              freq=self.freq,
                              scale_factor=self.scale_factor,
                              **self.params)
        return self.update(**kwargs)

    def update(self,
               dtype: Optional[Union[Type[np.float64],
                                     Type[np.complex128]]] = None,
               aos: Optional[str] = None,
               dof: Optional[int] = None,
               prev: Optional[ndarray] = None,
               phys: Optional[Physics] = None,
               freq: Optional[ndarray] = None,
               scale_factor: Optional[ndarray] = None,
               **params):
        dtype = self.dtype if dtype is None else dtype
        aos = self.aos if aos is None else aos
        dof = self.dof if dof is None else dof
        prev = self.prev if prev is None else prev
        phys = self.phys if phys is None else phys
        freq = self.freq if freq is None else freq
        scale_factor = self.scale_factor if scale_factor is None \
            else scale_factor
        return type(self)(form=self.form,
                          dtype=dtype,
                          aos=aos,
                          dof=dof,
                          prev=prev,
                          phys=phys,
                          freq=freq,
                          scale_factor=scale_factor,
                          **{**self.params, **params})

    def assemble_parameters(self):
        if self.scale_factor is None:
            scale_factor = np.array([1.] * self.dof)
        else:
            scale_factor = self.scale_factor
        return {
            'dof': self.dof,
            'phys': self.phys,
            'freq': self.freq,
            'scale_factor': scale_factor,
            **self.params
        }


class BilinearForm(Form):
    def _assemble(self,
                  ubasis: Basis,
                  vbasis: Optional[Basis] = None,
                  **kwargs) -> COOData:
        if vbasis is None:
            vbasis = ubasis
        else:
            raise NotImplementedError

        ne = ubasis.nelems
        udict = [FormExtraParams(x) for x in ubasis.basis]
        vdict = [FormExtraParams(x) for x in vbasis.basis]
        wdict = FormExtraParams({
            **ubasis.mesh_parameters(aos=self.aos),
            **ubasis.field_parameters(prev=self.prev, dof=self.dof),
            **self.assemble_parameters(),
            **kwargs
        })

        # initialize COO data structure
        dof = self.dof
        sz = dof * dof * ubasis.nverts * vbasis.nverts * ne
        data = np.zeros((dof, dof, ubasis.nverts, vbasis.nverts, ne),
                        dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int32)
        cols = np.zeros(sz, dtype=np.int32)

        for m in range(dof):
            for k in range(dof):
                for j in range(ubasis.nverts):
                    for i in range(vbasis.nverts):
                        start = ne * (dof * ubasis.nverts * vbasis.nverts * m +
                                      ubasis.nverts * vbasis.nverts * k +
                                      vbasis.nverts * j + i)
                        end = ne * (dof * ubasis.nverts * vbasis.nverts * m +
                                    ubasis.nverts * vbasis.nverts * k +
                                    vbasis.nverts * j + i + 1)
                        ixs = slice(start, end)
                        rows[ixs] = vbasis.gconns[i, :] * dof + k
                        cols[ixs] = ubasis.gconns[j, :] * dof + m
                        wdict['kron'] = 1 - np.abs(np.sign(i - j))
                        wdict['row'] = k
                        wdict['col'] = m
                        data[m, k, j, i, :] = self._kernel(
                            udict[j],
                            vdict[i],
                            wdict
                        )
        data = data.flatten('C')
        return COOData(
            np.array([rows, cols]),
            data,
            (vbasis.nnodes * dof,
             ubasis.nnodes * dof)
        )

    def _kernel(self, u, v, w):
        return self.form(u, v, w)

    def assemble(self, *args, **kwargs) -> Any:
        print("Assembling '{}'.".format(self.form.__name__))
        start = time.time()
        out = self._assemble(*args, **kwargs).todefault()
        elapsed = time.time() - start
        print("Assembling finished in {} seconds.".format(elapsed))
        return out


class LinearForm(Form):
    def _assemble(self,
                  ubasis: Basis,
                  vbasis: Optional[Basis] = None,
                  **kwargs):
        if vbasis is None:
            vbasis = ubasis
        else:
            raise NotImplementedError

        ne = ubasis.nelems
        vdict = [FormExtraParams(x) for x in vbasis.basis]
        wdict = FormExtraParams({
            **ubasis.mesh_parameters(aos=self.aos),
            **ubasis.field_parameters(prev=self.prev, dof=self.dof),
            **self.assemble_parameters(),
            **kwargs
        })

        # initialize COO data structure
        dof = self.dof
        sz = dof * vbasis.nverts * ne
        data = np.zeros((dof, vbasis.nverts, ne),
                        dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int32)

        for k in range(dof):
            for i in range(vbasis.nverts):
                start = ne * (vbasis.nverts * k + i)
                end = ne * (vbasis.nverts * k + i + 1)
                ixs = slice(start, end)
                rows[ixs] = vbasis.gconns[i, :] * dof + k
                wdict['kron'] = [1 - np.abs(np.sign(i - j))
                                 for j in range(vbasis.nverts)]
                wdict['row'] = k
                wdict['col'] = 0
                data[k, i, :] = self._kernel(vdict[i], wdict)
        data = data.flatten('C')
        return COOData(
            np.array([rows]),
            data,
            (vbasis.nnodes * dof,)
        )

    def _kernel(self, v, w):
        return self.form(v, w)

    def assemble(self, *args, **kwargs) -> Any:
        print("Assembling '{}'.".format(self.form.__name__))
        start = time.time()
        out = self._assemble(*args, **kwargs).todefault()
        elapsed = time.time() - start
        print("Assembling finished in {} seconds.".format(elapsed))
        return out
