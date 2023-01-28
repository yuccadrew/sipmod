import os

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from scipy.sparse import coo_matrix
from sipmod.utils import COOData, Physics
from sipmod.utils import save_dict, load_dict, amend_dict


class TestCOODataMatrixAddition(TestCase):
    def runTest(self):
        row = np.array([0, 3, 1, 2, 3, 2])
        col = np.array([0, 1, 1, 2, 0, 1])
        data = np.array([10, 3, 88, 9, 2, 6])

        A = coo_matrix((data, (row, col)), shape=(5, 4))
        A1 = COOData(np.vstack((row, col)), data, shape=(5, 4))
        A2 = COOData(np.vstack((row, col)), data, shape=(4, 4))
        assert_array_equal(A.toarray()*2, (A1+A2).toarray())


class TestPhysics(TestCase):
    def runTest(self):
        p = Physics(T=293., reps_w=80., reps_i=4.5,
                    c=np.array([1, 1]),
                    z=np.array([-1, 1]),
                    m_w=np.array([5e-8, 5e-8]),
                    s=np.array([0., 0.01]),
                    p=np.array([1, -1]),
                    m_s=np.array([5e-9, 0]))
        for ky in ['nions', 'kT', 'eps_i', 'eps_w', 'C', 'Q', 'D_s', 'D_w',
                   'kappa', 'debye_length', 'sigma_i', 'sigma_s', 'sigma_w']:
            assert getattr(p, ky) is not None
        assert_allclose(p.debye_length, 9.627066398783066e-09)


class TestDictIO(TestCase):
    def runTest(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            hf_prefix = 'tests/test_utils'
            # test save_dict for various dtype
            data1 = {
                'ndarray': np.arange(10, dtype=np.complex128),
                'string': 'apple',
                'int': 1,
                'float': 1.0,
                'complex': 1j,
                'bool': True,
                'none': None,
                'list': [1]
            }
            save_dict(hf_prefix, data1)
            data2 = load_dict(hf_prefix)
            for ky in data2:
                if isinstance(data2[ky], np.ndarray):
                    assert_array_equal(data1[ky], data2[ky])
                    assert_array_equal(data1[ky], load_dict(hf_prefix, ky))
                else:
                    assert data1[ky] == data2[ky]
                    assert data1[ky] == load_dict(hf_prefix, ky)

            # test amend_dict for existing key
            array1 = np.zeros((8, 10), dtype=np.complex128)
            for i in range(8):
                array1[i, :] = np.arange(10) + i
                amend_dict(hf_prefix, {'ndarray': array1})
            array2 = load_dict(hf_prefix)['ndarray']
            assert_array_equal(array1, array2)
            assert_array_equal(array1, load_dict(hf_prefix, 'ndarray'))

            # test amend_dict for new key
            amend_dict(hf_prefix, {'new_array': array1})
            assert_array_equal(array1, load_dict(hf_prefix, 'new_array'))

            # test amend_dict for replacing a row
            array2 = array1.copy()
            array2[-1, :] = array1[-1, :] + 1
            replace_row = array2.shape[0] - 1
            amend_dict(hf_prefix, {'new_array': array2[-1]}, replace_row)
            assert_array_equal(array2, load_dict(hf_prefix, 'new_array'))
            os.remove(hf_prefix+'.h5')
        MPI.Comm.Barrier(comm)
