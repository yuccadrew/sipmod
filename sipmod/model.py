import numpy as np
from scipy import constants

from .kernel import BilinearForm, LinearForm
from .mesh import Mesh
from .utils import Physics


class Poisson:
    @staticmethod
    def assemble(m: Mesh, p: Physics, **kwargs):
        from .kernel import CellBasisTri

        @BilinearForm
        def laplace(u, v, w):
            return (
                w.c * u.grad[0] * v.grad[0] +
                w.c * u.grad[1] * v.grad[1]
            ) * w.area * w.r

        bw = CellBasisTri(m, 'water')
        bs = CellBasisTri(m, 'solid')
        ba = CellBasisTri(m, 'air')

        A = (laplace(c=p.reps_i, aos=kwargs.get('aos', 'none')).assemble(bs) +
             laplace(c=p.reps_w, aos=kwargs.get('aos', 'none')).assemble(bw) +
             laplace(c=1.0, aos=kwargs.get('aos', 'none')).assemble(ba))
        b = np.zeros(A.shape[0])
        return A, b

    @staticmethod
    def dirichlet(m: Mesh, p: Physics, **kwargs):
        from .utils import COOData
        D = []
        s = []

        nodes = m.subdomain_nodes('default', reverse=True)
        D += [nodes['id']]
        s += [np.zeros(nodes['x'].shape[0])]

        nodes = m.boundary_nodes('outer')
        D += [nodes['id']]
        s += [np.zeros(nodes['x'].shape[0])]

        nodes = m.boundary_nodes('ep')
        D += [nodes['id']]
        s += [np.ones(nodes['x'].shape[0])*p.u_0]

        s, D = np.hstack(s), np.hstack(D)
        dirichlet = COOData(np.array([D]), s, (m.nnodes,))
        D = np.unique(D)
        s = dirichlet.toarray()[D]
        return s, D


class PoissonBoltzmann:
    @staticmethod
    def assemble(m: Mesh, p: Physics, **kwargs):
        from .kernel import CellBasisTri, FacetBasisTri

        @BilinearForm
        def laplace(u, v, w):
            return (
                w.c * u.grad[0] * v.grad[0] +
                w.c * u.grad[1] * v.grad[1]
            ) * w.area * w.r

        @BilinearForm
        def water_lhs(u, v, w):
            p = w.phys
            c = p.eps_w
            a = np.zeros_like(w.u)
            for i in range(p.nions):
                factor = p.Q[i] / p.kT
                a[:] += p.Q[i] * p.C[i] * np.exp(-factor * w.u) * (factor)
            return (
                c * u.grad[0] * v.grad[0] +
                c * u.grad[1] * v.grad[1] +
                a * (1 + w.kron) / 12
            ) * w.area * w.r

        @LinearForm
        def water_rhs(v, w):
            p = w.phys
            f = np.zeros_like(w.u)
            for i in range(p.nions):
                factor = p.Q[i] / p.kT
                f[:] += p.Q[i] * p.C[i] * np.exp(factor * w.u) * (factor * w.u
                                                                  - 1)
            return (f / 3) * w.area * w.r

        @LinearForm
        def stern_rhs(v, w):
            p = w.phys
            g = -p.sigma_w  # check the sign
            return (g / 2) * w.dx * w.r

        params = {
            'prev': np.zeros(m.nnodes),
            'phys': p,
            **kwargs
        }
        bw = CellBasisTri(m, 'water')
        bi = CellBasisTri(m, 'solid')
        ba = CellBasisTri(m, 'air')
        bs = FacetBasisTri(m, 'stern')
        if p.water_only:
            A = water_lhs(**params).assemble(bw)
        else:
            A = (
                water_lhs(**params).assemble(bw) +
                laplace(**params, c=p.eps_i).assemble(bi) +
                laplace(**params, c=p.eps_a).assemble(ba)
            )
        b = (
            water_rhs(**params).assemble(bw) +
            stern_rhs(**params).assemble(bs)
        )
        return A, b

    @staticmethod
    def dirichlet(m: Mesh, p: Physics, **kwargs):
        from .utils import COOData
        D = []
        s = []

        nodes = m.subdomain_nodes('default', reverse=True)
        D += [nodes['id']]
        s += [np.zeros(nodes['x'].shape[0])]

        nodes = m.boundary_nodes('outer')
        D += [nodes['id']]
        s += [np.zeros(nodes['x'].shape[0])]

        if p.water_only:
            nodes = m.subdomain_nodes('water', reverse=True)
            D += [nodes['id']]
            s += [np.zeros(nodes['x'].shape[0])]

        s, D = np.hstack(s), np.hstack(D)
        dirichlet = COOData(np.array([D]), s, (m.nnodes,))
        D = np.unique(D)
        s = dirichlet.toarray()[D]
        return s, D


class PoissonNernstPlanck:
    @staticmethod
    def assemble(m: Mesh, p: Physics, **kwargs):
        from .kernel import CellBasisTri, FacetBasisTri

        @BilinearForm
        def laplace(u, v, w):
            p = w.phys
            c = [[0.0] * w.dof for i in range(w.dof)]
            c[p.nions][p.nions] = w.c
            return (
                c[w.row][w.col] * u.grad[0] * v.grad[0] +
                c[w.row][w.col] * u.grad[1] * v.grad[1]
            ) * w.area * w.r * w.scale_factor[w.row]

        @BilinearForm
        def water_lhs(u, v, w):
            p = w.phys
            pot = w.u.reshape(-1, w.dof)[:, p.nions]
            dpot = w.grad.reshape(2, -1, w.dof)[:, :, p.nions]
            c = [[0.] * w.dof for i in range(w.dof)]
            a_x = [[0.] * w.dof for i in range(w.dof)]
            a_y = [[0.] * w.dof for i in range(w.dof)]
            a = [[0.] * w.dof for i in range(w.dof)]
            c[p.nions][p.nions] = p.eps_w
            for i in range(p.nions):
                # diagonal components
                c[i][i] = p.D_w[i]
                a_x[i][i] = p.z[i] * p.m_w[i] * dpot[0, :]
                a_y[i][i] = p.z[i] * p.m_w[i] * dpot[1, :]
                a[i][i] = 2j * np.pi * w.freq if w.freq is not None else 0
                # off-diagonal components
                factor = p.Q[i] / p.kT
                c[i][p.nions] = p.z[i] * p.m_w[i] * p.c[i] * np.exp(-factor
                                                                    * pot)
                a[p.nions][i] = - constants.e * constants.N_A * p.z[i]
            return (
                c[w.row][w.col] * u.grad[0] * v.grad[0] +
                c[w.row][w.col] * u.grad[1] * v.grad[1] +
                a_x[w.row][w.col] * v.grad[0] / 3 +
                a_y[w.row][w.col] * v.grad[1] / 3 +
                a[w.row][w.col] * (1 + w.kron) / 12
            ) * w.area * w.r * w.scale_factor[w.row]

        @BilinearForm
        def stern_lhs(u, v, w):
            p = w.phys
            # pot = w.u.reshape(-1, w.dof)[:, p.nions]
            dpot = w.grad.reshape(2, -1, w.dof)[:, :, p.nions]
            c = [[0.] * w.dof for i in range(w.dof)]
            a_x = [[0.] * w.dof for i in range(w.dof)]
            a = [[0.] * w.dof for i in range(w.dof)]
            q = [[0.] * w.dof for i in range(w.dof)]
            for i in range(p.nions+1, w.dof):
                # diagonal components
                c[i][i] = p.D_s[0]
                a_x[i][i] = p.p[0] * p.m_s[0] * dpot[0, :]
                a[i][i] = 2j * np.pi * w.freq if w.freq is not None else 0
                # off-diagonal components
                c[i][p.nions] = p.m_s[0] * p.sigma_s  # !!!take care
                q[p.nions][i] = -1 * p.p[0]  # check the sign
            return (
                c[w.row][w.col] * u.grad[0] * v.grad[0] +
                a_x[w.row][w.col] * v.grad[0] / 2 +
                a[w.row][w.col] * (1 + w.kron) / 6 +
                q[w.row][w.col] * (1 + w.kron) / 6
            ) * w.dx * w.r * w.scale_factor[w.row]

        @LinearForm
        def stern_rhs(v, w):
            p = w.phys
            g = [0.] * w.dof
            g[p.nions] = -p.sigma_w
            return (
                g[w.row] / 2
            ) * w.dx * w.r * w.scale_factor[w.row]

        if 'basis' in kwargs.keys():
            basis = kwargs.pop('basis')
            bw = basis['water']
            bi = basis['solid']
            ba = basis['air']
            bs = basis['stern']
        else:
            bw = CellBasisTri(m, 'water')
            bi = CellBasisTri(m, 'solid')
            ba = CellBasisTri(m, 'air')
            bs = FacetBasisTri(m, 'stern')
        dof = kwargs.get('dof', p.nions+2)
        params = {
            'dof': dof,
            'prev': np.zeros(m.nnodes*dof),
            'phys': p,
            **kwargs
        }
        if p.eliminate_sigma:
            if p.water_only:
                A = water_lhs(**params).assemble(bw)
            else:
                A = (
                    water_lhs(**params).assemble(bw) +
                    laplace(**params, c=p.eps_i).assemble(bi) +
                    laplace(**params, c=p.eps_a).assemble(ba)
                )
            if p.steady_state:
                b = stern_rhs(**params).assemble(bs)
            else:
                b = np.zeros(m.nnodes*dof)
        else:
            if p.water_only:
                A = (
                    water_lhs(**params).assemble(bw) +
                    stern_lhs(**params).assemble(bs)
                )
            else:
                A = (
                    water_lhs(**params).assemble(bw) +
                    stern_lhs(**params).assemble(bs) +
                    laplace(**params, c=p.eps_i).assemble(bi) +
                    laplace(**params, c=p.eps_a).assemble(ba)
                )
            b = np.zeros(m.nnodes*dof)
        return A, b

    @staticmethod
    def dirichlet(m: Mesh, p: Physics, **kwargs):
        from .utils import COOData
        dof = kwargs.get('dof', p.nions+2)
        D = []
        s = []

        nodes = m.subdomain_nodes('default', reverse=True)
        D += [nodes['id'] * dof + i for i in range(dof)]
        s += [np.zeros(nodes['x'].shape[0]) for i in range(dof)]

        nodes = m.subdomain_nodes('water', reverse=True)
        D += [nodes['id'] * dof + i for i in range(p.nions)]
        s += [np.zeros(nodes['x'].shape[0]) for i in range(p.nions)]

        nodes = m.boundary_nodes('stern', reverse=True)
        D += [nodes['id'] * dof + i for i in range(p.nions+1, dof)]
        s += [np.zeros(nodes['x'].shape[0]) for i in range(p.nions+1, dof)]

        nodes = m.boundary_nodes('inner')
        D += [nodes['id'] * dof + i for i in range(p.nions, p.nions+1)]
        s += [np.ones(nodes['x'].shape[0])*p.u_0
              for i in range(p.nions, p.nions+1)]  # !!! don't use dof-1

        if p.perfect_conductor:
            nodes = m.boundary_nodes('stern')
            D += [nodes['id'] * dof + i for i in range(p.nions, dof)]
            s += [np.zeros(nodes['x'].shape[0]) for i in range(p.nions, dof)]

        if p.steady_state:
            nodes = m.outer_boundary_nodes('default')
            D += [nodes['id'] * dof + i for i in range(dof)]
            s += [np.zeros(nodes['x'].shape[0]) for i in range(dof)]

            nodes = m.outer_boundary_nodes('water')
            D += [nodes['id'] * dof + i for i in range(p.nions)]
            s += [np.zeros(nodes['x'].shape[0]) + p.c[i]
                  for i in range(p.nions)]

            nodes = m.outer_boundary_nodes('stern')
            D += [nodes['id'] * dof + i for i in range(p.nions+1, dof)]
            s += [np.zeros(nodes['x'].shape[0]) - p.sigma_w
                  for i in range(p.nions+1, dof)]
        else:
            nodes = m.outer_boundary_nodes('default')
            D += [nodes['id'] * dof + i for i in range(dof)]
            s += [np.zeros(nodes['x'].shape[0]) for i in range(dof)]

            nodes = m.outer_boundary_nodes('default')
            D += [nodes['id'] * dof + i for i in range(p.nions, p.nions+1)]
            s += [-nodes['x'] * p.e_0[0] - nodes['y'] * p.e_0[1]
                  for i in range(p.nions, p.nions+1)]  # !!! don't use dof-1

        if p.eliminate_conc:
            nodes = m.subdomain_nodes('water')
            D += [nodes['id'] * dof + i for i in range(p.nions)]
            if p.steady_state:
                s += [np.zeros(nodes['x'].shape[0]) + p.c[i]
                      for i in range(p.nions)]
            else:
                s += [np.zeros(nodes['x'].shape[0]) for i in range(p.nions)]

        if p.eliminate_sigma:
            nodes = m.boundary_nodes('stern')
            D += [nodes['id'] * dof + i for i in range(p.nions+1, dof)]
            if p.steady_state and not p.perfect_conductor:
                s += [np.zeros(nodes['x'].shape[0]) - p.sigma_w
                      for i in range(p.nions+1, dof)]
            else:
                s += [np.zeros(nodes['x'].shape[0])
                      for i in range(p.nions+1, dof)]

        if p.water_only:
            nodes = m.subdomain_nodes('water', reverse=True)
            D += [nodes['id'] * dof + i for i in range(p.nions, dof)]
            s += [np.zeros(nodes['x'].shape[0]) for i in range(p.nions, dof)]

        if 's' in kwargs and 'D' in kwargs:
            D += [kwargs['D']]
            s += [kwargs['s']]

        s, D = np.hstack(s), np.hstack(D)
        tmp = COOData(np.array([D]), s, (m.nnodes*dof,))
        D = np.unique(D)
        s = tmp.toarray()[D]
        if 's' in kwargs and 'D' in kwargs:
            tmp = COOData(np.array([kwargs['D']]),
                          kwargs['s'],
                          (m.nnodes*dof,))
            D2 = np.unique(kwargs['D'])
            s2 = tmp.toarray()[D2]
            mask = np.in1d(D, D2, assume_unique=True)
            s[mask] = s2
        return s, D
