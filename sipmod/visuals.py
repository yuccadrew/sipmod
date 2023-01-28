"""Drawing meshes and solutions using matplotlib."""

from functools import singledispatch

from numpy import ndarray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable

from .mesh import MeshTri, MpartTri
from .kernel import CellBasisTri, FacetBasisTri
from .utils import PolyData
# plt.style.use('seaborn-poster')


@singledispatch
def draw(m, **kwargs) -> Axes:
    """Visualize meshes."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@draw.register(MeshTri)
def draw_meshtri(m: MeshTri, **kwargs) -> Axes:
    """
    Visualize a two-dimensional triangular mesh by drawing the edges.

    Parameters
    ----------
    m
        A two-dimensional triangular mesh.
    ax (optional)
        A preinitialized Matplotlib axes for plotting.
    aspect (optional)
        Ratio of vertical to horizontal length-scales.
    color (optional)
        Color of the edges.
    linewidth (optional)
        Width of the edges.
    boundaries_only (optional)
        If ``True``, draw only boundary edges.

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    """
    if 'ax' not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
    else:
        ax = kwargs['ax']

    color = kwargs['color'] if 'color' in kwargs else 'tab:blue'
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 0.2

    x = m.nodes['x']
    y = m.nodes['y']
    triangles = m.elements['lconns'].T
    ax.triplot(x, y, triangles, color=color, linewidth=linewidth)
    if m.axis_of_symmetry.lower() in ['x']:
        ax.triplot(x, -y, triangles, color=color, linewidth=linewidth)
    elif m.axis_of_symmetry.lower() in ['y']:
        ax.triplot(-x, y, triangles, color=color, linewidth=linewidth)

    ax.show = lambda: plt.show()
    return ax


@draw.register(MpartTri)
def draw_mparttri(m: MpartTri, **kwargs) -> Axes:
    if 'ax' not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
    else:
        ax = kwargs['ax']

    color = kwargs['color'] if 'color' in kwargs else 'tab:blue'
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 0.2

    x = m.nodes['x']
    y = m.nodes['y']
    triangles = m.elements['lconns'][:, ~m.elements['ghost']].T
    ax.triplot(x, y, triangles, color=color, linewidth=linewidth)
    if m.axis_of_symmetry.lower() in ['x']:
        ax.triplot(x, -y, triangles, color=color, linewidth=linewidth)
    elif m.axis_of_symmetry.lower() in ['y']:
        ax.triplot(-x, y, triangles, color=color, linewidth=linewidth)

    ax.show = lambda: plt.show()
    return ax


@draw.register(CellBasisTri)
def draw_cbasistri(m: CellBasisTri, **kwargs) -> Axes:
    if 'ax' not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
    else:
        ax = kwargs['ax']

    color = kwargs['color'] if 'color' in kwargs else 'tab:blue'
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 0.2

    x = m._mesh.nodes['x']
    y = m._mesh.nodes['y']
    triangles = m.lconns.T
    ax.triplot(x, y, triangles, color=color, linewidth=linewidth)
    ax.show = lambda: plt.show()
    return ax


@draw.register(FacetBasisTri)
def draw_fbasistri(m: FacetBasisTri, **kwargs) -> Axes:
    if 'ax' not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
    else:
        ax = kwargs['ax']

    color = kwargs['color'] if 'color' in kwargs else 'tab:blue'
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 0.2

    xverts = m.xverts
    yverts = m.yverts
    ax.plot(xverts, yverts, color=color, linewidth=linewidth)
    ax.show = lambda: plt.show()
    return ax


@draw.register(PolyData)
def draw_PolyData(m: PolyData, **kwargs) -> Axes:
    if 'ax' not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
    else:
        ax = kwargs['ax']

    color = kwargs['color'] if 'color' in kwargs else 'tab:blue'
    linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else 0.2

    xverts = m.xverts
    yverts = m.yverts
    ax.plot(xverts, yverts, color=color, linewidth=linewidth)
    ax.show = lambda: plt.show()
    return ax


@singledispatch
def plot(m, u, **kwargs) -> Axes:
    """Plot functions defined on nodes of the mesh."""
    raise NotImplementedError("Type {} not supported.".format(type(m)))


@plot.register(MeshTri)
def plot_meshtri(m: MeshTri, z: ndarray, **kwargs):
    """
    Visualize a piece-wise linear function defined on a triangular mesh.

    Parameters
    ----------
    m
        A two-dimensional triangular mesh.
    ax (optional)
        A preinitialized Matplotlib axes for plotting.
    aspect (optional)
        Ratio of vertical to horizontal length-scales.
    figsize (optional)
        Passed on to matplotlib.
    logscale (optional)
        If True, show color map in logscale. Shown in linear scale by default.
    colorbar (optional)
        If True, show colorbar. If a string, use it as label for the colorbar.
        Shown by default.
    contours (optional)
        If True, show contour lines. Not shown by default.
    shading (optional)
    cmap (optional)
    vmin (optional)
    vmax (optional)
    edgecolors (optional)
    levels (optional)
    colors (optional)

    Returns
    -------
    Axes
        The Matplotlib axes onto which the mesh was plotted.

    """
    if 'ax' not in kwargs:
        fig = plt.figure(**{k: v for k, v in kwargs.items()
                            if k in ['figsize']})
        ax = fig.add_subplot(111)
        aspect = kwargs["aspect"] if "aspect" in kwargs else 1.0
        ax.set_aspect(aspect)
    else:
        ax = kwargs['ax']

    plot_kwargs = {
        'logscale': False,
        'colorbar': True,
        'contours': False,
        'shading': 'flat',  # or 'gouraud'
        'cmap': 'YlGnBu_r',
        'vmin': None,
        'vmax': None,
        'edgecolors': 'none',
        'levels': None,
        'colors': 'w',
        'scale_factor': 1.0,
        'cbtitle': None,
        **kwargs
    }

    cnorm = mcolors.LogNorm if plot_kwargs['logscale'] else mcolors.Normalize
    plot_kwargs['norm'] = cnorm(plot_kwargs['vmin'], plot_kwargs['vmax'])

    x = m.nodes['x'] * plot_kwargs['scale_factor']
    y = m.nodes['y'] * plot_kwargs['scale_factor']
    triangles = m.elements['lconns'].T
    ax.tripcolor(x, y, triangles, z,
                 **{k: v for k, v in plot_kwargs.items()
                    if k in ['shading',
                             'cmap',
                             'norm',
                             'edgecolors']})
    if plot_kwargs['colorbar']:
        cb = plt.colorbar(ScalarMappable(norm=plot_kwargs['norm'],
                                         cmap=plot_kwargs['cmap']), ax=ax)
        if isinstance(plot_kwargs['cbtitle'], str):
            cb.set_label(plot_kwargs['cbtitle'], rotation=90)

    if m.axis_of_symmetry.lower() in ['x']:
        ax.tripcolor(x, -y, triangles, z,
                     **{k: v for k, v in plot_kwargs.items()
                        if k in ['shading',
                                 'cmap',
                                 'norm',
                                 'edgecolors']})
    elif m.axis_of_symmetry.lower() in ['y']:
        ax.tripcolor(-x, y, triangles, z,
                     **{k: v for k, v in plot_kwargs.items()
                        if k in ['shading',
                                 'cmap',
                                 'norm',
                                 'edgecolors']})

    if plot_kwargs['contours']:
        ax.tricontour(x, y, triangles, z,
                      **{k: v for k, v in plot_kwargs.items()
                         if k in ['levels',
                                  'colors']})
        if m.axis_of_symmetry.lower() in ['x']:
            ax.tricontour(x, -y, triangles, z,
                          **{k: v for k, v in plot_kwargs.items()
                             if k in ['levels',
                                      'colors']})
        elif m.axis_of_symmetry.lower() in ['y']:
            ax.tricontour(-x, y, triangles, z,
                          **{k: v for k, v in plot_kwargs.items()
                             if k in ['levels',
                                      'colors']})

    ax.show = lambda: plt.show()
    return ax
