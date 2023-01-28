import importlib
import time

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Callable, Tuple

import numpy as np
from numpy import ndarray

from .mpart import read_my_nodes
from .mpart import read_my_elements
from .mpart import allocate_AIJ_sparse


def build_topological_edges(facets: ndarray) -> ndarray:
    """Return lower dimensional sorted edges given facets."""
    edges = np.hstack((
        facets[[0, 1], :],
        facets[[1, 2], :],
        facets[[2, 0], :],
    ))
    sorted_edges = np.unique(np.sort(edges, axis=0), axis=1)
    return sorted_edges


def read_triangle_mesh(mesh_prefix: str) -> Tuple[Dict[str, ndarray],
                                                  Dict[str, ndarray],
                                                  Dict[str, ndarray]]:
    locs = np.genfromtxt(
        mesh_prefix+'.node', skip_header=1, usecols=(1, 2)
    )
    flags = np.genfromtxt(
        mesh_prefix+'.node', skip_header=1, usecols=(3), dtype=np.int32
    )
    nodes = {
        'x': locs[:, 0],
        'y': locs[:, 1],
        'flags': flags,
        'id': np.arange(locs.shape[0], dtype=np.int32),
        'ghost': np.zeros(locs.shape[0], dtype=bool)
    }

    conns = np.genfromtxt(
        mesh_prefix+'.ele', skip_header=1, usecols=(1, 2, 3), dtype=np.int32
    ).T - 1  # zero-based indexing
    flags = np.genfromtxt(
        mesh_prefix+'.ele', skip_header=1, usecols=(4), dtype=np.int32
    )
    elems = {
        'gconns': conns,
        'lconns': conns,
        'flags': flags,
    }

    conns = np.genfromtxt(
        mesh_prefix+'.edge', skip_header=1, usecols=(1, 2), dtype=np.int32
    ).T - 1  # zero-based indexing
    flags = np.genfromtxt(
        mesh_prefix+'.edge', skip_header=1, usecols=(3), dtype=np.int32
    )
    edges = {
        'gconns': conns,
        'lconns': conns,
        'flags': flags
    }
    return nodes, elems, edges


def map_global_to_local(local2global: ndarray,
                        glob_conns: ndarray) -> ndarray:
    """Map global nodal indices to local nodal indices."""
    nnodes = np.maximum(np.max(glob_conns.ravel()), np.max(local2global)) + 1
    global2local = np.zeros(nnodes, dtype=np.int32) - 1
    global2local[local2global] = np.arange(local2global.shape[0])
    loc_conns = global2local[glob_conns]
    return loc_conns


def find_my_elements(mesh_prefix: str,
                     glob_nid: ndarray) -> Dict[str, ndarray]:
    """Read elements from file and stip extra elements."""
    gconns = np.genfromtxt(
        mesh_prefix+'.ele', skip_header=1, usecols=(1, 2, 3), dtype=np.int32
    ).T - 1  # zero-based indexing
    flags = np.genfromtxt(
        mesh_prefix+'.ele', skip_header=1, usecols=(4), dtype=np.int32
    )
    mask = np.zeros(gconns.shape, dtype=bool)
    for i in range(3):
        mask[i, :] = np.in1d(gconns[i, :], glob_nid)  # !!!take care
    mask = mask.all(axis=0)
    gconns = gconns[:, mask]
    flags = flags[mask]
    lconns = map_global_to_local(glob_nid, gconns)
    return {
        'gconns': gconns,
        'lconns': lconns,
        'flags': flags,
    }


def find_my_edges(mesh_prefix: str,
                  glob_nid: ndarray) -> Dict[str, ndarray]:
    """Read edges from file and strip extra edges."""
    gconns = np.genfromtxt(
        mesh_prefix+'.edge', skip_header=1, usecols=(1, 2), dtype=np.int32
    ).T - 1  # zero-based indexing
    flags = np.genfromtxt(
        mesh_prefix+'.edge', skip_header=1, usecols=(3), dtype=np.int32
    )
    mask = np.zeros(gconns.shape, dtype=bool)
    for i in range(2):
        mask[i, :] = np.in1d(gconns[i, :], glob_nid)  # !!!take care
    mask = mask.all(axis=0)
    gconns = gconns[:, mask]
    flags = flags[mask]
    lconns = map_global_to_local(glob_nid, gconns)
    return {
        'gconns': gconns,
        'lconns': lconns,
        'flags': flags,
    }


class DomainFlags:
    """Flags of mesh subdomains."""
    SOLID: ClassVar[int] = 1
    WATER: ClassVar[int] = 2
    AIR: ClassVar[int] = 3


class BoundaryFlags:
    """Flags of mesh boundaries."""
    SW: ClassVar[int] = 1  # solid-water interface
    AW: ClassVar[int] = 2  # air-water interface
    EP: ClassVar[int] = 3  # equipotential surface
    AOS: ClassVar[int] = 4  # axis of symmetry

    # sequences to be updated as left/right/bottom/top as 1/2/3/4
    LEFT: ClassVar[int] = 13  # left boundary
    RIGHT: ClassVar[int] = 14  # right boundary
    BOTTOM: ClassVar[int] = 12  # bottom boundary
    TOP: ClassVar[int] = 11  # topmost boundary


@dataclass(repr=False)
class Mesh:
    """A finite element mesh."""
    nodes: Dict[str, ndarray]  # location of the finite element nodes
    elements: Dict[str, ndarray]  # connectivity of the elements
    facets: Dict[str, ndarray]  # connectivity of the facets
    edges: Dict[str, ndarray]  # connectivity of the edges
    axis_of_symmetry: str = 'none'  # x or y or none
    scale_factor: float = 1.0  # distance scaling factor
    _nnodes: Optional[int] = None  # total number of nodes among all processors
    _aij_args: Optional[Dict[str, ndarray]] = None  # arguments for PETScSolver
    _subdomains: Optional[Dict[str, ndarray]] = None  # named subdomains
    _boundaries: Optional[Dict[str, ndarray]] = None  # named boundaries

    @property
    def nnodes(self):
        return self._nnodes

    @property
    def nverts(self):
        return self.elements['lconns'].shape[0]

    @property
    def nelements(self):
        return self.elements['lconns'].shape[1]

    @property
    def nfacets(self):
        return self.facets['lconns'].shape[1]

    @property
    def nedges(self):
        return self.edges['lconns'].shape[1]

    @property
    def subdomains(self):
        """Return named subdomains in boolean arrays."""
        return self._subdomains

    @property
    def boundaries(self):
        """Return named boundaries in boolean arrays."""
        return self._boundaries

    @property
    def x(self):
        return self.nodes['x']

    @property
    def y(self):
        return self.nodes['y']

    @property
    def z(self):
        if self.dim() == 3:
            return self.nodes['z']
        else:
            raise NotImplementedError

    def aij_args(self,
                 dof: int = 1,
                 nonghost_rows: bool = True,
                 nonghost_cols: bool = False,
                 unique_indices: bool = False,
                 test_indices: bool = False) -> Dict[str, ndarray]:
        prows = []
        pcols = []
        if test_indices:
            for m in range(dof):
                for k in range(dof):
                    for j in range(self.nverts):
                        for i in range(self.nverts):
                            for e in range(self.nelements):
                                loc_nid = self.elements['lconns'][i, e]
                                if not self.nodes['ghost'][loc_nid]:
                                    r = self.elements['gconns'][i, e] * dof + k
                                    c = self.elements['gconns'][j, e] * dof + m
                                    prows.append(r)
                                    pcols.append(c)
        else:
            nonghost = ~self.nodes['ghost'][self.elements['lconns']]
            for m in range(dof):
                for k in range(dof):
                    for j in range(self.nverts):
                        for i in range(self.nverts):
                            mask = np.ones(self.nelements, dtype=bool)
                            if nonghost_rows:
                                mask = mask & nonghost[i, :]
                            if nonghost_cols:
                                mask = mask & nonghost[j, :]
                            r = self.elements['gconns'][i, mask] * dof + k
                            c = self.elements['gconns'][j, mask] * dof + m
                            prows.append(r)
                            pcols.append(c)
        prows = np.hstack(prows).astype(np.int32)
        pcols = np.hstack(pcols).astype(np.int32)
        if unique_indices:
            indices = np.unique(np.array([prows, pcols]), axis=1)
        else:
            indices = np.array([prows, pcols])

        if dof > 1:
            raise NotImplementedError
        else:
            return {
                **self._aij_args,
                'indices': indices,
            }

    def dim(self):
        raise NotImplementedError

    def subdomain_nodes(self,
                        name: str = 'default',
                        reverse: bool = False) -> Dict[str, ndarray]:
        """Return named subdomain nodes."""
        mask = self.subdomains[name] if name in self.subdomains else []
        loc_nid = np.unique(self.elements['lconns'][:, mask].ravel())
        if reverse:
            tmp = np.ones(self.nodes['x'].shape[0], dtype=bool)
            tmp[loc_nid] = False
            loc_nid = np.where(tmp)[0]
        return {k: v[loc_nid] for k, v in self.nodes.items()}

    def boundary_nodes(self,
                       name: str = 'default',
                       reverse: bool = False) -> Dict[str, ndarray]:
        """Return named boundary nodes."""
        mask = self.boundaries[name] if name in self.boundaries else []
        loc_nid = np.unique(self.facets['lconns'][:, mask].ravel())
        if reverse:
            tmp = np.ones(self.nodes['x'].shape[0], dtype=bool)
            tmp[loc_nid] = False
            loc_nid = np.where(tmp)[0]
        return {k: v[loc_nid] for k, v in self.nodes.items()}

    def inner_boundary_nodes(self,
                             name: str = 'default',
                             reverse: bool = False) -> Dict[str, ndarray]:
        pass

    def outer_boundary_nodes(self,
                             name: str = 'default',
                             reverse: bool = False) -> Dict[str, ndarray]:
        """Return named outer boundary nodes."""
        mask = self.boundaries['outer']
        loc_nid = np.unique(self.facets['lconns'][:, mask].ravel())
        if name in ['outer', 'default']:
            tmp = loc_nid
        elif name in self.subdomains:
            mask = self.subdomains[name]
            tmp = np.unique(self.elements['lconns'][:, mask].ravel())
        elif name in self.boundaries:
            mask = self.boundaries[name]
            tmp = np.unique(self.facets['lconns'][:, mask].ravel())
        else:
            tmp = []
        idx = np.intersect1d(loc_nid, tmp, return_indices=True)[1]
        if reverse:
            tmp = np.ones(loc_nid.shape[0], dtype=bool)
            tmp[idx] = False
            loc_nid = loc_nid[tmp]
        else:
            loc_nid = loc_nid[idx]
        return {k: v[loc_nid] for k, v in self.nodes.items()}

    def subdomain_elements(self, name: str = 'default') -> Dict[str, ndarray]:
        """Return named subdomain elements."""
        mask = self.subdomains[name] if name in self.subdomains else []
        lconns = self.elements['lconns'][:, mask]
        return {
            **{k: v[:, mask] for k, v in self.elements.items()
               if k in ['gconns', 'lconns']},
            **{k + 'verts': v[lconns] for k, v in self.nodes.items()
               if k in ['x', 'y', 'z']}
        }

    def boundary_elements(self, name: str = 'default') -> Dict[str, ndarray]:
        mask = self.boundaries[name] if name in self.boundaries else []
        lconns = self.facets['lconns'][:, mask]
        return {
            **{k: v[:, mask] for k, v in self.facets.items()
               if k in ['gconns', 'lconns']},
            **{k + 'verts': v[lconns] for k, v in self.nodes.items()
               if k in ['x', 'y', 'z']}
        }

    def with_subdomains(self, name: Dict[str, Callable[[ndarray], ndarray]]):
        """Return a copy of the mesh with named subdomains."""
        raise NotImplementedError

    def with_boundaries(self, name: Dict[str, Callable[[ndarray], ndarray]]):
        """Return a copy of the mesh with named boundaries."""
        raise NotImplementedError

    def build_topological_facets(self):
        raise NotImplementedError

    def build_topological_edges(self):
        raise NotImplementedError

    def _init_subdomains(self):
        """Initialize ``self._subdomains``."""
        subdomains = {}
        for ky in ['solid', 'water', 'air']:
            subdomains[ky] = self.elements['flags'] == \
                getattr(DomainFlags, ky.upper())
        subdomains['default'] = (subdomains['solid'] | subdomains['water']) | \
            subdomains['air']
        self._subdomains = subdomains

    def _init_boundaries(self):
        """Initialize ``self._boundaries``."""
        boundaries = {}
        for ky in ['sw', 'aw', 'ep', 'aos', 'left', 'right', 'bottom', 'top']:
            boundaries[ky] = self.facets['flags'] == \
                getattr(BoundaryFlags, ky.upper())
        boundaries['default'] = boundaries['sw']
        boundaries['robin'] = boundaries['sw']
        boundaries['stern'] = boundaries['sw']
        boundaries['inner'] = boundaries['ep']
        boundaries['outer'] = (boundaries['left'] | boundaries['right']) | \
            (boundaries['bottom'] | boundaries['top'])
        self._boundaries = boundaries

    def __post_init__(self):
        """Scale the nodal coordinates etc."""
        if 'x' in self.nodes:
            self.nodes['x'][:] = self.nodes['x']*self.scale_factor
        if 'y' in self.nodes:
            self.nodes['y'][:] = self.nodes['y']*self.scale_factor
        if 'z' in self.nodes:
            self.nodes['z'][:] = self.nodes['z']*self.scale_factor

        if self._nnodes is None:
            self._nnodes = self.nodes['x'].shape[0]
        if self._aij_args is None:
            self._aij_args = {}
        if self._subdomains is None:
            self._init_subdomains()
        if self._boundaries is None:
            self._init_boundaries()

    def __repr__(self):
        rep = ""
        rep += "<{} object>\n".format(type(self).__name__)
        rep += "  Number of nodes: {}\n".format(self.nnodes)
        rep += "  Number of elements: {}\n".format(self.nelements)
        rep += "  Number of facets: {}\n".format(self.nfacets)
        rep += "  Number of edges: {}".format(self.nedges)
        rep += "\n  Named subdomains [# elements]: {}".format(
            ', '.join(
                map(lambda k: '{} [{}]'.format(k, np.sum(self.subdomains[k])),
                    list(self.subdomains.keys()))
            )
        )
        if self.dim() == 3:
            rep += "\n  Named boundaries [# facets]: {}".format(
                ', '.join(
                    map(lambda k: '{} [{}]'.format(
                        k, np.sum(self.boundaries[k])),
                        list(self.boundaries.keys()))
                )
            )
        else:
            rep += "\n  Named boundaries [# edges]: {}".format(
                ', '.join(
                    map(lambda k: '{} [{}]'.format(
                        k, np.sum(self.boundaries[k])),
                        list(self.boundaries.keys()))
                )
            )
        return rep

    def __str__(self):
        return self.__repr__()

    def save(self):
        raise NotImplementedError

    @classmethod
    def load(self):
        raise NotImplementedError

    def draw(self, **kwargs):
        """Convenience wrapper for sipmod.visuals."""
        mod = importlib.import_module('sipmod.visuals')
        return mod.draw(self, **kwargs)

    def plot(self, u: ndarray, **kwargs):
        """Convenience wrapper for sipmod.visuals."""
        mod = importlib.import_module('sipmod.visuals')
        return mod.plot(self, u, **kwargs)


@dataclass(repr=False)
class MeshTri(Mesh):
    """A 2D triangular mesh."""

    @classmethod
    def read(cls, mesh_prefix: str, **kwargs):
        """Read 2D mesh files generated by triangle and return mesh."""
        print("Initializing {}('{}')".format(cls.__name__, mesh_prefix))
        start = time.time()
        if not mesh_prefix[-2:] in ['.1', '.2']:
            mesh_prefix = mesh_prefix+'.1'

        nodes, elems, edges = read_triangle_mesh(mesh_prefix)
        elapsed = time.time() - start
        print("Initializing finished in {} seconds.".format(elapsed))
        return cls(
            nodes=nodes,
            elements=elems,
            facets=edges,  # alias for edges in 2D
            edges=edges,
            **kwargs
        )

    def dim(self):
        return 2


@dataclass(repr=False)
class MpartTri(Mesh):
    """A 2D partitioned triangular mesh."""

    @classmethod
    def read(cls, mesh_prefix: str, myrank: int = 0, **kwargs):
        """Read 2D partitioned mesh files and return mesh."""
        print("Initializing {}('{}')".format(cls.__name__, mesh_prefix))
        start = time.time()
        if not mesh_prefix.endswith('.2'):
            mesh_prefix = mesh_prefix+'.2'

        dof = 1
        my_elems, ghost_elems = read_my_elements(mesh_prefix, myrank)
        nnodes, nstart, nend, my_nodes = read_my_nodes(
            mesh_prefix, myrank, dof, my_elems, ghost_elems)
        aij_args = allocate_AIJ_sparse(
            nnodes, nstart, nend, dof, my_nodes, my_elems, ghost_elems)

        nodes = {
            'x': np.zeros(len(my_nodes)),
            'y': np.zeros(len(my_nodes)),
            'flags': np.zeros(len(my_nodes), dtype=np.int32),
            'id': np.zeros(len(my_nodes), dtype=np.int32),
            'ghost': np.zeros(len(my_nodes), dtype=bool),
        }
        for i in range(len(my_nodes)):
            nodes['x'][i] = my_nodes[i].x
            nodes['y'][i] = my_nodes[i].y
            nodes['flags'][i] = my_nodes[i].flags[0]
            nodes['id'][i] = my_nodes[i].g_ind
            nodes['ghost'][i] = my_nodes[i].ghost

        all_elems = my_elems + ghost_elems
        elems = {
            'gconns': np.zeros((3, len(all_elems)), dtype=np.int32),
            'lconns': np.zeros((3, len(all_elems)), dtype=np.int32),
            'flags': np.zeros(len(all_elems), dtype=np.int32),
        }
        for i in range(len(all_elems)):
            elems['gconns'][:, i] = all_elems[i].g_nodes
            elems['lconns'][:, i] = all_elems[i].l_nodes
            elems['flags'][i] = all_elems[i].flags[0]

        edges = find_my_edges(mesh_prefix, nodes['id'])
        elapsed = time.time() - start
        print("Initializing finished in {} seconds.".format(elapsed))
        return cls(
            nodes=nodes,
            # elements=find_my_elements(mesh_prefix, nodes['id']),
            elements=elems,
            facets=edges,  # alias for edges in 2D
            edges=edges,
            _nnodes=nnodes,
            _aij_args=aij_args,
            **kwargs
        )

    def dim(self):
        return 2
