from mpi4py import MPI
from unittest import TestCase
import pytest

import numpy as np
from numpy.testing import assert_array_equal

from sipmod.mesh import MeshTri, read_triangle_mesh
from sipmod.mesh import MpartTri, map_global_to_local
from sipmod.mesh import build_topological_edges
from sipmod.mpart import metis_partition_tri
from sipmod.mpart import read_my_elements
from sipmod.mpart import read_my_nodes


class TestReadTriangleMesh(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            read_triangle_mesh(mesh_prefix+'.1')
        MPI.Comm.Barrier(comm)


class TestMeshTriRead(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            MeshTri.read(mesh_prefix+'.1')
        MPI.Comm.Barrier(comm)


class TestMpartTriRead(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)
        MpartTri.read(mesh_prefix+'.2', comm.Get_rank())


class TestMeshAttributes(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)

        if comm.Get_size() == 1:
            mesh1 = MeshTri.read(mesh_prefix+'.2')
            mesh2 = MpartTri.read(mesh_prefix+'.2', comm.Get_rank())
            for attr in ['nodes', 'elements', 'facets', 'edges',
                         'subdomains', 'boundaries']:
                dict1 = getattr(mesh1, attr)
                dict2 = getattr(mesh2, attr)
                assert dict1.keys() == dict2.keys()
                for ky in dict1:
                    assert dict1[ky] is not None
                    assert_array_equal(dict1[ky], dict2[ky])


class TestSubdomainNodes(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)

        if comm.Get_size() == 1:
            mesh1 = MeshTri.read(mesh_prefix+'.2')
            mesh2 = MpartTri.read(mesh_prefix+'.2', comm.Get_rank())
            assert mesh1.subdomains.keys() == mesh2.subdomains.keys()
            for method in ['subdomain_nodes']:
                for name in list(mesh1.subdomains) + ['none']:
                    dict1 = getattr(mesh1, method)(name)
                    dict2 = getattr(mesh2, method)(name)
                    assert dict1.keys() == dict2.keys()
                    for ky in dict1:
                        assert dict1[ky] is not None
                        assert dict2[ky] is not None
                        assert_array_equal(dict1[ky], dict2[ky])
                        if name in ['none']:
                            assert dict1[ky].shape[0] == 0
                            assert dict2[ky].shape[0] == 0


class TestBoundaryNodes(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)

        if comm.Get_size() == 1:
            mesh1 = MeshTri.read(mesh_prefix+'.2')
            mesh2 = MpartTri.read(mesh_prefix+'.2', comm.Get_rank())
            assert mesh1.boundaries.keys() == mesh2.boundaries.keys()
            for method in ['boundary_nodes']:
                for name in list(mesh1.boundaries) + ['none']:
                    dict1 = getattr(mesh1, method)(name)
                    dict2 = getattr(mesh2, method)(name)
                    assert dict1.keys() == dict2.keys()
                    for ky in dict1:
                        assert dict1[ky] is not None
                        assert dict2[ky] is not None
                        if ky in ['xverts', 'yverts', 'lconns', 'gconns']:
                            assert dict1[ky].ndim == 2
                            assert dict1[ky].shape[0] == 3
                            assert dict1[ky].shape[1] == \
                                dict1['xverts'].shape[1]
                        assert_array_equal(dict1[ky], dict2[ky])
                        if name in ['none']:
                            assert dict1[ky].shape[0] == 0
                            assert dict2[ky].shape[0] == 0


class TestSubdomainElements(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)

        if comm.Get_size() == 1:
            mesh1 = MeshTri.read(mesh_prefix+'.2')
            mesh2 = MpartTri.read(mesh_prefix+'.2', comm.Get_rank())
            assert mesh1.subdomains.keys() == mesh2.subdomains.keys()
            for method in ['subdomain_elements']:
                for name in list(mesh1.subdomains) + ['none']:
                    dict1 = getattr(mesh1, method)(name)
                    dict2 = getattr(mesh2, method)(name)
                    assert dict1.keys() == dict2.keys()
                    for ky in dict1:
                        assert dict1[ky] is not None
                        assert dict2[ky] is not None
                        if ky in ['xverts', 'yverts', 'lconns', 'gconns']:
                            assert dict1[ky].ndim == 2
                            assert dict1[ky].shape[0] == 3
                            assert dict1[ky].shape[1] == \
                                dict1['xverts'].shape[1]
                        assert_array_equal(dict1[ky], dict2[ky])
                        if name in ['none']:
                            assert dict1[ky].shape[-1] == 0
                            assert dict2[ky].shape[-1] == 0


class TestBoundaryElements(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)

        if comm.Get_size() == 1:
            mesh1 = MeshTri.read(mesh_prefix+'.2')
            mesh2 = MpartTri.read(mesh_prefix+'.2', comm.Get_rank())
            assert mesh1.subdomains.keys() == mesh2.subdomains.keys()
            self.assertEqual(mesh1.boundaries.keys(), mesh2.boundaries.keys())
            for method in ['boundary_elements']:
                for name in list(mesh1.boundaries) + ['none']:
                    dict1 = getattr(mesh1, method)(name)
                    dict2 = getattr(mesh2, method)(name)
                    assert dict1.keys() == dict2.keys()
                    for ky in dict1:
                        assert dict1[ky] is not None
                        assert dict2[ky] is not None
                        assert_array_equal(dict1[ky], dict2[ky])
                        if name in ['none']:
                            assert dict1[ky].shape[-1] == 0
                            assert dict2[ky].shape[-1] == 0


class TestGlobalToLocalMapping(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            dof = 1
            npart = 3
            metis_partition_tri(mesh_prefix.split('.')[0], npart,
                                clean_files=True)
            for mypart in range(npart):
                my_elems, ghost_elems = read_my_elements(
                    mesh_prefix+'.2', mypart
                )
                nnodes, nstart, nend, nodes = read_my_nodes(
                    mesh_prefix+'.2', mypart, dof, my_elems, ghost_elems
                )

                g_inds = np.zeros(len(nodes), dtype=int)
                for i in range(len(nodes)):
                    g_inds[i] = nodes[i].g_ind

                for e in my_elems + ghost_elems:
                    g_nodes = e.g_nodes
                    l_nodes = e.l_nodes
                    t_nodes = map_global_to_local(g_inds, g_nodes)
                    assert_array_equal(l_nodes, t_nodes)
        MPI.Comm.Barrier(comm)


class TestTopologicalEdges(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_box'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            metis_partition_tri(mesh_prefix, comm.Get_size())
        MPI.Comm.Barrier(comm)

        mesh = MpartTri.read(mesh_prefix+'.2', comm.Get_rank())
        sorted_edges1 = build_topological_edges(mesh.elements['lconns'])
        sorted_edges2 = np.unique(
            np.sort(mesh.edges['lconns'], axis=0), axis=1)
        assert_array_equal(sorted_edges1, sorted_edges2)
        assert mesh.nedges == sorted_edges2.shape[1]


class TestAijArgsFromMeshTriRead(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_ex03'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            from sipmod.kernel import CellBasisTri, BilinearForm

            @BilinearForm
            def laplace(u, v, w):
                return (
                    u.grad[0] * v.grad[0] +
                    u.grad[1] * v.grad[1]
                ) * w.area

            nrank = 8
            metis_partition_tri(mesh_prefix, nrank)
            mesh = MeshTri.read(mesh_prefix+'.2')
            basis = CellBasisTri(mesh)
            indices0 = laplace._assemble(basis).indices
            indices1 = mesh.aij_args(nonghost_rows=False)['indices']
            assert_array_equal(indices0, indices1)
        MPI.Comm.Barrier(comm)


class TestAijArgsFromMpartTriRead(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_ex03'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            from sipmod.kernel import CellBasisTri, BilinearForm

            @BilinearForm
            def laplace(u, v, w):
                return (
                    u.grad[0] * v.grad[0] +
                    u.grad[1] * v.grad[1]
                ) * w.area

            nrank = 8
            metis_partition_tri(mesh_prefix, nrank)
            for myrank in range(nrank):
                mesh = MpartTri.read(mesh_prefix+'.2', myrank)
                basis = CellBasisTri(mesh)
                indices0 = laplace._assemble(basis).indices
                indices1 = mesh.aij_args(nonghost_rows=False)['indices']
                assert_array_equal(indices0, indices1)
        MPI.Comm.Barrier(comm)


class TestAijArgsWithoutGhostRows(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_ex03'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            nrank = 8
            metis_partition_tri(mesh_prefix, nrank)
            for myrank in range(nrank):
                mesh = MpartTri.read(mesh_prefix+'.2', myrank)
                indices0 = mesh.aij_args(test_indices=True)['indices']
                indices1 = mesh.aij_args(test_indices=False)['indices']
                assert_array_equal(indices0, indices1)
        MPI.Comm.Barrier(comm)


class TestPartitionedNonGhostNodes(TestCase):
    def runTest(self):
        mesh_prefix = 'docs/examples/meshes/mesh_ex03'
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            nrank = 8
            metis_partition_tri(mesh_prefix, nrank)
            ms = MeshTri.read(mesh_prefix+'.2')
            glob_nid = []
            for myrank in range(nrank):
                mp = MpartTri.read(mesh_prefix+'.2', myrank)
                glob_nid.append(mp.nodes['id'][~mp.nodes['ghost']])
            assert ms.nnodes == np.hstack(glob_nid).shape[0]
            assert ms.nnodes == np.unique(np.hstack(glob_nid)).shape[0]
        MPI.Comm.Barrier(comm)


@pytest.mark.skip(reason="xfailed due to bugs in ghost elements/nodes")
def test_partitioned_ghost_nodes():
    mesh_prefix = 'docs/examples/meshes/mesh_ex03'
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        nrank = 8
        metis_partition_tri(mesh_prefix, nrank)
        ms = MeshTri.read(mesh_prefix+'.2')
        aij_args1 = ms.aij_args(nonghost_rows=False)
        for myrank in range(nrank):
            mp = MpartTri.read(mesh_prefix+'.2', myrank)
            aij_args2 = mp.aij_args(nonghost_rows=False)
            glob_nid = mp.nodes['id'][~mp.nodes['ghost']]
            for idx in glob_nid:
                mask = aij_args1['indices'][0, :] == idx
                scols = np.unique(aij_args1['indices'][1, :][mask])
                mask = aij_args2['indices'][0, :] == idx
                pcols = np.unique(aij_args2['indices'][1, :][mask])
                assert_array_equal(scols, pcols)
    MPI.Comm.Barrier(comm)
