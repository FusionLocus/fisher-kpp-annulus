import adios4dolfinx
from dolfinx import fem
from basix.ufl import element
from ufl import SpatialCoordinate, inner, as_vector, grad
import configparser
import numpy as np
from mpi4py import MPI
from simutils.postproc import prepare_objects, get_indices_isoline
from simutils.vis import set_size
import matplotlib.pyplot as plt
import matplotlib as mpl
from isoline_analysis_delta_r0 import isoline_properties_single_simulation, plot_all_metrics


if __name__ == '__main__':      
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    config = configparser.ConfigParser()
    config.read('/home/fenicsx/fisher-kpp-annulus/realisations/parameter-sweep-kbar-epsilon-higherres/config.ini')
    time = 2.5

    r0_min = config.getfloat('param.r0', 'minimum')
    r0_max = config.getfloat('param.r0', 'maximum')
    n_r0 = config.getint('param.r0', 'n')

    r0_vals = np.linspace(r0_min, r0_max, num=n_r0)

    k_min = config.getfloat('param.k', 'minimum')
    k_max = config.getfloat('param.k', 'maximum')
    n_k = config.getint('param.k', 'n')

    k_vals = np.logspace(k_min, k_max, num=n_k)
    print(k_vals*config.getfloat('geometry', 'delta')**2 /config.getfloat('param', 'D'))
    kbar_vals = k_vals*config.getfloat('geometry', 'delta')**2 /config.getfloat('param', 'D')
    delta_val = config.getfloat('geometry', 'delta')

    all_norms = np.empty((n_r0, n_k))
    all_norms[:] = np.nan
    dtheta = np.empty((n_r0, n_k))
    dtheta[:] = np.nan
    dtheta_norm = np.empty((n_r0, n_k))
    dtheta_norm[:] = np.nan

    epsilons = np.empty((n_r0, n_k))
    epsilons[:] = np.nan
    drtheta = np.empty((n_r0, n_k))
    drtheta[:] = np.nan

    dspeed = np.empty((n_r0, n_k))
    dspeed[:] = np.nan

    n_isopoints = []

    for r0_idx, r0_val in enumerate(r0_vals):
        for k_idx, k_val in enumerate(k_vals):
            r0_k_str = f'r0={r0_val:.2f}, k={k_val:.2f}'
            folder_path_arc = config['sim']['output_dir_main'][:-1] + '-arcs/' + r0_k_str + '/'
            iso_coords_arc = isoline_properties_single_simulation(config, folder_path_arc, time=2.5)
            if np.round(r0_val - delta_val/2, 4) <= 0:
                    print(f'Skipping: No longer annulus geometry with ' + r0_k_str)
                    all_norms[r0_idx, k_idx] = np.nan

            else:
                config['param']['k'] = str(k_val)
                config['geometry']['r0'] = str(r0_val)

                folder_path = config['sim']['output_dir_main'] + r0_k_str + '/'

                iso_coords = isoline_properties_single_simulation(config, folder_path, time=time)
                if iso_coords is None:
                    print('No coordinates found.')
                    continue
                dot_prod_rms = np.power(np.sum(np.power(iso_coords['nabla_u'], 2))/ iso_coords['nabla_u'].size, 0.5)

                max_idx, min_idx = np.argmax(iso_coords['r']), np.argmin(iso_coords['r'])
                print(r0_val, delta_val, np.max(iso_coords['th']), np.min(iso_coords['th']))

                dtheta[r0_idx, k_idx] = np.abs(np.max(iso_coords['th']) - np.min(iso_coords['th']))
                
                all_norms[r0_idx, k_idx] = dot_prod_rms
                epsilons[r0_idx, k_idx] = delta_val / r0_val
                drtheta[r0_idx, k_idx] = np.abs(iso_coords['r'][max_idx]*iso_coords['th'][max_idx] \
                                                    - iso_coords['r'][min_idx]*iso_coords['th'][min_idx])

                dspeed[r0_idx, k_idx] = np.abs(np.mean(iso_coords['tang_velocity']/iso_coords['r']) - np.mean(iso_coords_arc['tang_velocity'])/r0_val)

    quantities = [all_norms, dtheta, dspeed, epsilons]
    #labels = ['Curvature penalisation', r'$|\theta_{\delta/2} - \theta_{-\delta/2}$',
    #          r'$|\langle \omega_i \rangle - \omega_{r_0}|$', r'$\epsilon$']
    labels = [r'$S_1$', r'$S_2$', r'$S_3$', r'$\epsilon$']

    plot_all_metrics(quantities, labels, r0_vals, np.log10(k_vals),
                     x_label=r'$r_0$', y_label=r'$\log_{10}\bar{k}$')
    
    """
    fig, axs = plt.subplots(figsize=set_size('thesis'))
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = axs.imshow(all_norms.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # #for i, r0 in enumerate(r0_vals[:-1]):
    #     for j, k in enumerate(k_vals[:-1]):
    #         if not np.isnan(all_norms.T[i, j]):
    #             text = axs.text(j, i, np.round(all_norms.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    plt.xlabel(r'$r_0$')
    #plt.xlim([0.1, 2.10])
    
    plt.xticks(ticks= np.arange(0, r0_vals.size, step = 4), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::4]])
    #plt.ylim([0.1, 2.10])
    plt.ylabel(r'$\log_{10}\bar{k}$')
    plt.yticks(ticks= np.arange(0, k_vals.size, step = 4),
                labels=[fr'${np.log10(k):.2f}$' for k in kbar_vals[::2]])
    plt.colorbar(im)
    fig.suptitle('Curvature penalisation')
    plt.savefig('curvature_penalisation.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(figsize=set_size('thesis'))
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = axs.imshow(dtheta.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # for i, r0 in enumerate(r0_vals[:-1]):
    #     for j, k in enumerate(k_vals[:-1]):
    #         if not np.isnan(dtheta.T[i, j]):
    #             text = axs.text(j, i, np.round(dtheta.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    
    fig.suptitle(r'$|\theta_{\delta/2} - \theta_{-\delta/2}|$')
    plt.xlabel(r'$r_0$')
    #plt.xlim([0.1, 2.10])
    
    plt.xticks(ticks= np.arange(0, r0_vals.size, step = 4), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::4]])
    #plt.ylim([0.1, 2.10])
    plt.ylabel(r'$\log_{10}\bar{k}$')
    plt.yticks(ticks= np.arange(0, k_vals.size, step = 4),
                labels=[fr'${np.log10(k):.2f}$' for k in kbar_vals[::2]])
    plt.colorbar(im)
    plt.savefig('theta_diff.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(figsize=set_size('thesis'))
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = axs.imshow(drtheta.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # for i, r0 in enumerate(r0_vals[:-1]):
    #     for j, k in enumerate(k_vals[:-1]):
    #         if not np.isnan(drtheta.T[i, j]):
    #             text = axs.text(j, i, np.round(drtheta.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    
    fig.suptitle(r'$|\text{max}(r_i \theta_i) - \text{min}(r_i \theta_i)|$')
    plt.xlabel(r'$r_0$')
    #plt.xlim([0.1, 2.10])
    
    plt.xticks(ticks= np.arange(0, r0_vals.size, step = 4), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::4]])
    #plt.ylim([0.1, 2.10])
    plt.ylabel(r'$\log_{10}\bar{k}$')
    plt.yticks(ticks= np.arange(0, k_vals.size, step = 4),
                labels=[fr'${np.log10(k):.2f}$' for k in kbar_vals[::2]])
    plt.colorbar(im)
    plt.savefig('drtheta.pdf', bbox_inches='tight')


    fig, axs = plt.subplots(figsize=set_size('thesis'))
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = axs.imshow(epsilons.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # for i, r0 in enumerate(r0_vals[:-1]):
    #     for j, k in enumerate(k_vals[:-1]):
    #         if not np.isnan(epsilons.T[i, j]):
    #             text = axs.text(j, i, np.round(epsilons.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    
    fig.suptitle(r'$\epsilon$')
    plt.xlabel(r'$r_0$')
    #plt.xlim([0.1, 2.10])
    
    plt.xticks(ticks= np.arange(0, r0_vals.size, step = 4), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::4]])
    #plt.ylim([0.1, 2.10])
    plt.ylabel(r'$\log_{10}\bar{k}$')
    plt.yticks(ticks= np.arange(0, k_vals.size, step = 4),
                labels=[fr'${np.log10(k):.2f}$' for k in kbar_vals[::2]])
    plt.colorbar(im)
    plt.savefig('epsilon.pdf', bbox_inches='tight')

    plt.show()

    fig, axs = plt.subplots()
    plt.hist(n_isopoints, bins=20)
    plt.savefig('iso_points_distribution.pdf')
    """

