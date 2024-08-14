import adios4dolfinx
from dolfinx import fem
from basix.ufl import element
import configparser
import numpy as np
from mpi4py import MPI
from simutils.vis import arrange_contour_plot
import matplotlib.pyplot as plt

filepath = './realisations/base-halfannulus/'

config = configparser.ConfigParser()
config.read(filepath + 'config.ini')

soln_file = filepath + 'soln.bp'

times = np.linspace(
    config.getfloat('sim.times', 't0'),
    config.getfloat('sim.times', 'tf'),
    int(config.getint('sim.times', 'n_t')/100)+1
)

msh = adios4dolfinx.read_mesh(soln_file, MPI.COMM_WORLD)
V = fem.functionspace(msh, element(config['sim']['element_type'], msh.basix_cell(), config.getint('sim.times', 'degree')))
coords = V.tabulate_dof_coordinates()
u = fem.Function(V)

x = coords[:, 0]
y = coords[:, 1]

fig, axs = plt.subplots(ncols=2)

if config['sim']['case'] == 'annulus':
    annulus=True

else:
    annulus=False

for t in times:

    adios4dolfinx.read_function(soln_file, u, time=np.round(t, 4), name='u')
    u_vals = u.x.array

    arrange_contour_plot(x, y, u_vals, axs[0], axs[1], 'test', annulus=annulus, plot_border=True, paper=True)
    fig.savefig(filepath+f'imgs/summary-{np.round(t, 4)}.pdf')
