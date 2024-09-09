import adios4dolfinx
from dolfinx import fem
from basix.ufl import element
import configparser
import numpy as np
from mpi4py import MPI
from simutils.vis import arrange_contour_plot
import matplotlib.pyplot as plt
from isoline_analysis_delta_r0 import isoline_properties_single_simulation

def produce_contour_plots(config, filepath):
    soln_file = filepath + 'soln.bp'

    times = np.linspace(
        config.getfloat('sim.times', 't0'),
        config.getfloat('sim.times', 'tf'),
        int(config.getint('sim.times', 'n_t')/100)+1
    )

    msh = adios4dolfinx.read_mesh(soln_file, MPI.COMM_WORLD)
    V = fem.functionspace(msh, element('DG', msh.basix_cell(), 0))
    coords = V.tabulate_dof_coordinates()
    u = fem.Function(V)

    x = coords[:, 0]
    y = coords[:, 1]

    fig, axs = plt.subplots(ncols=2)

    if config['sim']['case'] == 'annulus':
        annulus=True

    else:
        annulus=False

    for t in np.linspace(0, 12, num=25):

        adios4dolfinx.read_function(soln_file, u, time=np.round(t, 4), name='u')
        u_vals = u.x.array

        arrange_contour_plot(x, y, u_vals, axs[0], axs[1], 'test', annulus=annulus, plot_border=True, paper=True)
        fig.savefig(filepath+f'imgs/summary-{np.round(t, 4)}.pdf')

if __name__ == '__main__':

    filepath = './realisations/base-rectangle/' #kbar-epsilon/'
    config = configparser.ConfigParser()
    config.read(filepath + 'config.ini')

    #produce_contour_plots(config, filepath)
    for t in np.linspace(0, 12, num=25):
        coord_dict = isoline_properties_single_simulation(config, filepath, time=t, annulus=False)
        fig, ax = plt.subplots()
        ax.scatter(coord_dict['y'], coord_dict['x']-np.mean(coord_dict['x']), marker='x')
        plt.savefig(filepath + f'test-{t:.2f}.pdf')
        plt.close(fig)
