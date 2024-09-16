import adios4dolfinx
from dolfinx import fem
from basix.ufl import element
import configparser
import numpy as np
from mpi4py import MPI
from simutils.vis import arrange_contour_plot
import matplotlib.pyplot as plt
from isoline_analysis_delta_r0 import isoline_properties_single_simulation

def produce_contour_plots(config, filepath, snapshots=False):
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
    print(coords.shape)

    x = coords[:, 0]
    y = coords[:, 1]
    if config['ic']['func'] == 'gaussian':
        t_interrogate = [0.01, 0.5, 1.0, 2.0]
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6.5*0.5*2, 6.5*0.5*2))
        early_times = True
    elif snapshots:
        t_interrogate = [0.5, 1, 1.5, 2, 2.5, 3]
        early_times = False
    else:
        t_interrogate = [1.0, 3.0, 5.0]
        fig, axs = plt.subplots(ncols=3, figsize=(3.2, 0.728))
        early_times = False
    #cax = plt.axes((0.95, 0.1, 0.075, 0.8))
    
    if config['sim']['case'] == 'annulus' or config['sim']['case'] == 'threequarter':
        annulus=True

    else:
        annulus=False

    for idx, t in enumerate(t_interrogate):

        adios4dolfinx.read_function(soln_file, u, time=np.round(t, 4), name='u')
        u_vals = u.x.array
        if snapshots:
            fig, ax = plt.subplots()
            arrange_contour_plot(x, y, u_vals, ax, 'test', annulus=annulus, plot_border=True, paper=True,
                             early_times=early_times)
            fig.savefig(filepath+f'imgs/halfannulus-summary-{t:.2f}s.png', bbox_inches='tight', dpi=600)
        
        else:
            arrange_contour_plot(x, y, u_vals, axs.flat[idx], 'test', annulus=annulus, plot_border=True, paper=True,
                             early_times=early_times)
            axs.flat[idx].spines['top'].set_visible(False)
            axs.flat[idx].spines['right'].set_visible(False)

    
        
    if not snapshots:
        fig.savefig(filepath+f'imgs/halfannulus-summary-earlytimes.png', bbox_inches='tight', dpi=600)

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    filepath = './realisations/parameter-sweep-kbar-epsilon-higherres/r0-0.40,k-10/' #kbar-epsilon/'
    config = configparser.ConfigParser()
    config.read(filepath + 'config.ini')

    produce_contour_plots(config, filepath, snapshots=True)
    #for t in np.linspace(0, 12, num=25):
    #    coord_dict = isoline_properties_single_simulation(config, filepath, time=t, annulus=False)
    #    fig, ax = plt.subplots()
    #    ax.scatter(coord_dict['y'], coord_dict['x']-np.mean(coord_dict['x']), marker='x')
    #    plt.savefig(filepath + f'test-{t:.2f}.pdf')
    #    plt.close(fig)
