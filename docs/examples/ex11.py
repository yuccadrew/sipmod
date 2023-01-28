"""
Generate AFM probe mesh using Capsol (FD) nodal points.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np  # noqa
from sipmod import CapSolData  # noqa


inpargs = {'n': 500, 'm': 500, 'l_js': 500,
           'h0': 1.0, 'rho_max': 1e6, 'z_max': 1e6,
           'd_min': 100.0, 'd_max': 100.0, 'idstep': 2,
           'Rtip': 20.0, 'theta': 15.0,
           'HCone': 15e3, 'RCant': 35e3, 'dCant': 0.5e3,
           'reps_s': 5.0, 'Hsam': 1e6,
           'reps_w': 1.0, 'Hwat': 5.0,  # equivalent to air
           'd': 100.0}  # tip-sample separation in use
capsol = CapSolData(**inpargs)
mesh_prefix = 'docs/examples/meshes/mesh_ex11'


def run_capsol():
    path = os.path.dirname(mesh_prefix)
    capsol.write_inputs(path=path)  # write capsol.in
    os.system('bash {}/capsol_run.sh'.format(path))  # call capsol


def build_mesh():
    poly = capsol.generate_poly()
    poly.write(mesh_prefix)  # write poly file
    poly.build(mesh_prefix, flag='-pnAae')  # call triangle


def demo():
    import matplotlib.pyplot as plt
    from sipmod.visuals import draw
    from sipmod.mesh import MeshTri
    mesh = MeshTri.read(mesh_prefix)

    fig, ax = plt.subplots(2, 2)
    draw(mesh, ax=ax[0, 0])
    ax[0, 0].set_ylabel('Y (nanometer)')
    ax[0, 0].set_xlabel('X (nanometer)')
    ax[0, 0].set_title('Entire Domain')

    draw(mesh, ax=ax[0, 1])
    ax[0, 1].set_xlim(0, 42e3)
    ax[0, 1].set_ylim(0, 42e3)
    ax[0, 1].set_ylabel('Y (nanometer)')
    ax[0, 1].set_xlabel('X (nanometer)')
    ax[0, 1].set_title('Probe')

    draw(mesh, ax=ax[1, 0])
    ax[1, 0].set_xlim(0, 50)
    ax[1, 0].set_ylim(0, 50)
    ax[1, 0].set_ylabel('Y (nanometer)')
    ax[1, 0].set_xlabel('X (nanometer)')
    ax[1, 0].set_title('Tip')

    draw(mesh, ax=ax[1, 1])
    ax[1, 1].set_xlim(0, 50)
    ax[1, 1].set_ylim(-125, -75)
    ax[1, 1].set_ylabel('Y (nanometer)')
    ax[1, 1].set_xlabel('X (nanometer)')
    ax[1, 1].set_title('Layers')
    plt.tight_layout()
    return plt


def visualize():
    import matplotlib.pyplot as plt
    from sipmod.visuals import draw
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.8))
    poly = capsol.generate_poly()
    draw(poly, linewidth=1.0, ax=ax[0, 0])
    ax[0, 0].set_ylabel('Y (nanometer)')
    ax[0, 0].set_xlabel('X (nanometer)')
    ax[0, 0].set_title('Entire Domain')

    draw(poly, linewidth=1.0, ax=ax[0, 1])
    ax[0, 1].set_xlim(0, 42e3)
    ax[0, 1].set_ylim(0, 42e3)
    ax[0, 1].set_ylabel('Y (nanometer)')
    ax[0, 1].set_xlabel('X (nanometer)')
    ax[0, 1].set_title('Probe')

    draw(poly, linewidth=1.0, ax=ax[1, 0])
    ax[1, 0].set_xlim(0, 50)
    ax[1, 0].set_ylim(0, 50)
    ax[1, 0].set_ylabel('Y (nanometer)')
    ax[1, 0].set_xlabel('X (nanometer)')
    ax[1, 0].set_title('Tip')

    draw(poly, linewidth=1.0, ax=ax[1, 1])
    ax[1, 1].set_xlim(0, 50)
    ax[1, 1].set_ylim(-125, -75)
    ax[1, 1].set_ylabel('Y (nanometer)')
    ax[1, 1].set_xlabel('X (nanometer)')
    ax[1, 1].set_title('Layers')
    plt.tight_layout()

    from sipmod.mesh import MeshTri
    mesh = MeshTri.read(mesh_prefix)
    fig, ax = plt.subplots(2, 2)
    draw(mesh, ax=ax[0, 0])
    ax[0, 0].set_ylabel('Y (nanometer)')
    ax[0, 0].set_xlabel('X (nanometer)')
    ax[0, 0].set_title('Entire Domain')

    draw(mesh, ax=ax[0, 1])
    ax[0, 1].set_xlim(0, 42e3)
    ax[0, 1].set_ylim(0, 42e3)
    ax[0, 1].set_ylabel('Y (nanometer)')
    ax[0, 1].set_xlabel('X (nanometer)')
    ax[0, 1].set_title('Probe')

    draw(mesh, ax=ax[1, 0])
    ax[1, 0].set_xlim(0, 50)
    ax[1, 0].set_ylim(0, 50)
    ax[1, 0].set_ylabel('Y (nanometer)')
    ax[1, 0].set_xlabel('X (nanometer)')
    ax[1, 0].set_title('Tip')

    draw(mesh, ax=ax[1, 1])
    ax[1, 1].set_xlim(0, 50)
    ax[1, 1].set_ylim(-125, -75)
    ax[1, 1].set_ylabel('Y (nanometer)')
    ax[1, 1].set_xlabel('X (nanometer)')
    ax[1, 1].set_title('Layers')
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    run_capsol()  # comment this line if you don't have capsol compiled
    build_mesh()  # comment this line if you don't have triangle compiled
    visualize().show()
