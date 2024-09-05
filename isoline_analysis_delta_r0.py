import adios4dolfinx
from dolfinx import fem
from basix.ufl import element
from ufl import SpatialCoordinate, inner, as_vector, grad, div, sqrt
import configparser
import numpy as np
from mpi4py import MPI
from simutils.postproc import prepare_objects, get_indices_isoline
from simutils.vis import set_size
import matplotlib.pyplot as plt
import matplotlib as mpl


def isoline_properties_single_simulation(config, folder_path, time=None, annulus=True):

    coords, u, V, times, soln_file, msh = prepare_objects(config, folder_path)
    x = SpatialCoordinate(msh)
    if time is None:
        # Load in at all times. TBC
        pass

    else:

        coord_dict = {}
        adios4dolfinx.read_function(soln_file, u, time=np.round(time, 4), name="u")
        # nabla u hat dot xhat
        x_grad = inner(grad(u), as_vector((x[0]/(x[0]**2 + x[1]**2)**0.5, x[1]/(x[0]**2 + x[1]**2)**0.5, 0))) \
            / sqrt(inner(grad(u), grad(u)))
        
        # dudt/ |nabla u|**2 * nabla u
        speed = (div(grad(u)) + u*(1-u))/sqrt(inner(grad(u), grad(u)))
        tang_velocity = inner(speed *grad(u)/sqrt(inner(grad(u), grad(u))), as_vector(
            (-x[1]/(x[0]**2 + x[1]**2)**0.5, x[0]/(x[0]**2 + x[1]**2)**0.5, x[2])
        ))
        #W = fem.functionspace(msh, ('DG', 1))
        expr_u_DG = fem.Expression(u, V.element.interpolation_points())
        expr = fem.Expression(x_grad, V.element.interpolation_points())
        expr2 = fem.Expression(speed, V.element.interpolation_points())
        expr3 = fem.Expression(tang_velocity, V.element.interpolation_points())
        
        u_DG = fem.Function(V)
        w = fem.Function(V)
        w2 = fem.Function(V)
        w3 = fem.Function(V)

        u_DG.interpolate(expr_u_DG)
        w.interpolate(expr)
        w2.interpolate(expr2)
        w3.interpolate(expr3)

        u_vals = u_DG.x.array[:]
        if np.min(u_DG.x.array[:]) > 0.01:
            r0 = (np.max(coords[:, 0]) + np.min(coords[:, 0]))/2
            delta = np.max(coords[:, 0]) - np.min(coords[:, 0])
            print(np.min(u.x.array[:]), r0, delta)
        mask = np.abs(u_vals - 0.5) < 0.005

        iso_coords = coords[mask, :]
        coord_dict['x'], coord_dict['y'] = iso_coords[:, 0], iso_coords[:, 1]
        if annulus:
            iso_r = np.power(np.power(iso_coords[:, 0], 2) + np.power(iso_coords[:, 1], 2), 0.5)
            iso_th = np.arctan2(iso_coords[:, 1], iso_coords[:, 0])

            nabla_u = np.zeros((iso_r.size, 1))
            nabla_u[:, 0] = w.x.array[mask]
            
            speed = np.zeros((iso_r.size, 1))
            speed[:, 0] = w2.x.array[mask]

            vel = np.zeros((iso_r.size, 1))
            vel[:, 0] = w3.x.array[mask]

            coord_dict['r'], coord_dict['th'] = iso_r, iso_th
            coord_dict['nabla_u'] = nabla_u
            coord_dict['speed'] = speed
            coord_dict['tang_velocity'] = vel

        return coord_dict

def plot_all_metrics(quantities, labels, cmap=mpl.colormaps.get_cmap('inferno'), 
                     padding_factor=1.4, x_label='', y_label=''):
        cmap.set_bad(color='grey')
        fig, axs = plt.subplots(nrows=2, 
                                ncols=2, 
                                figsize=set_size('thesis', 
                                        subplots=(2, 2),
                                        padding_adjustment=padding_factor))
        
        for idx, (quantity, label) in enumerate(zip(quantities, labels)):
            if idx == 0:
                row, col = 0, 0
            elif idx == 1:
                row, col = 0, 1
            elif idx == 2:
                row, col = 1, 0
            elif idx == 3:
                row, col = 1, 1

            ax = axs[row, col]
            
            im = ax.imshow(quantity.T, 
                    interpolation='none',
                    cmap=cmap,
                    origin='lower',
                    resample=False)
        
            ax.set_xticks(ticks= np.arange(0, r0_vals.size, step = 2), 
                    labels=[fr'${r0:.1f}$' for r0 in r0_vals[::2]])

            ax.set_yticks(ticks= np.arange(0, delta_vals.size, step = 2), 
                    labels=[fr'${delta:.1f}$' for delta in delta_vals[::2]])
            cb = fig.colorbar(im, ax=ax)

            cb.locator = mpl.ticker.LinearLocator(numticks=3)
            cb.ax.ticklabel_format(style='sci', scilimits=(0, 0))
            cb.update_ticks()
            ax.title.set_text(label)

        fig.text(0.5, 0.04, x_label, ha='center')
        fig.text(0.04, 0.5, y_label, ha='center')
        plt.subplots_adjust(wspace=0.3, hspace=(padding_factor-1)/padding_factor)
        plt.savefig('all_metrics_delta_r0.pdf', bbox_inches='tight')
        return


if __name__ == '__main__':      
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    config = configparser.ConfigParser()
    config.read('./realisations/parameter-sweep-higherres/config.ini')
    time = 2.5000

    r0_min = config.getfloat('param.r0', 'minimum')
    r0_max = config.getfloat('param.r0', 'maximum')
    n_r0 = config.getint('param.r0', 'n')

    r0_vals = np.linspace(r0_min, r0_max, num=n_r0)

    delta_min = config.getfloat('param.delta', 'minimum')
    delta_max = config.getfloat('param.delta', 'maximum')
    n_delta = config.getint('param.delta', 'n')

    delta_vals = np.linspace(delta_min, delta_max, num=n_delta)

    all_norms = np.empty((n_r0, n_delta))
    all_norms[:] = np.nan
    dtheta = np.empty((n_r0, n_delta))
    dtheta[:] = np.nan
    epsilons = np.empty((n_r0, n_delta))
    epsilons[:] = np.nan
    drtheta = np.empty((n_r0, n_delta))
    drtheta[:] = np.nan

    dspeed = np.empty((n_r0, n_delta))
    dspeed[:] = np.nan

    for r0_idx, r0_val in enumerate(r0_vals):
        r0_arc_str = f'r0={r0_val:.2f}, delta=0.00'
        folder_path_arc = config['sim']['output_dir_main'][:-1] + '-arcs/' + r0_arc_str + '/'
        iso_coords_arc = isoline_properties_single_simulation(config, folder_path_arc, time=2.5)
        
        for delta_idx, delta_val in enumerate(delta_vals):
            r0_delta_str = f'r0={r0_val:.2f}, delta={delta_val:.2f}'
            if np.round(r0_val - delta_val/2, 4) <= 0:
                    print(f'Skipping: No longer annulus geometry with ' + r0_delta_str)
                    all_norms[r0_idx, delta_idx] = np.nan

            else:
                config['geometry']['delta'] = str(delta_val)
                config['geometry']['r0'] = str(r0_val)

                folder_path = config['sim']['output_dir_main'] + r0_delta_str + '/'

                iso_coords = isoline_properties_single_simulation(config, folder_path, time=2.5)
                dot_prod_rms = np.power(np.sum(np.power(iso_coords['nabla_u'], 2))/ iso_coords['nabla_u'].size, 0.5)

                max_idx, min_idx = np.argmax(iso_coords['r']), np.argmin(iso_coords['r'])
                print(r0_val, delta_val, np.max(iso_coords['th']), np.min(iso_coords['th']))

                dtheta[r0_idx, delta_idx] = np.abs(np.max(iso_coords['th']) - np.min(iso_coords['th']))
                
                all_norms[r0_idx, delta_idx] = dot_prod_rms
                epsilons[r0_idx, delta_idx] = delta_val / r0_val
                drtheta[r0_idx, delta_idx] = np.abs(iso_coords['r'][max_idx]*iso_coords['th'][max_idx] \
                                                    - iso_coords['r'][min_idx]*iso_coords['th'][min_idx])

                dspeed[r0_idx, delta_idx] = np.abs(np.mean(iso_coords['tang_velocity']/iso_coords['r']) - np.mean(iso_coords_arc['tang_velocity'])/r0_val)


    
    quantities = [all_norms, dtheta, dspeed, epsilons]
    #labels = ['Curvature penalisation', r'$|\theta_{\delta/2} - \theta_{-\delta/2}$',
    #          r'$|\langle \omega_i \rangle - \omega_{r_0}|$', r'$\epsilon$']
    labels = [r'$S_1$', r'$S_2$', r'$S_3$', r'$\epsilon$']
    
    plot_all_metrics(quantities, labels, 
                     x_label=r'$r_0$', y_label=r'$\delta$')


fig, ax = plt.subplots()

cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
cmap.set_bad(color='grey')
im = ax.imshow(drtheta.T, 
            interpolation='none',
            cmap=cmap,
            origin='lower',
            resample=False)

ax.title.set_text(r'$\text{max}(r_i\theta_i) - \text{min}(r_i\theta_i)$')


ax.set_xticks(ticks= np.arange(0, r0_vals.size, step = 2), 
            labels=[fr'${r0:.1f}$' for r0 in r0_vals[::2]])

ax.set_yticks(ticks= np.arange(0, delta_vals.size, step = 2), 
            labels=[fr'${delta:.1f}$' for delta in delta_vals[::2]])
fig.colorbar(im, ax=ax)
plt.savefig('drtheta_delta_r0.pdf')

"""
ax = axs[0, 1]
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = ax.imshow(dtheta.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # for i, r0 in enumerate(r0_vals):
    #     for j, delta in enumerate(delta_vals):
    #         if not np.isnan(dtheta.T[i, j]):
    #             text = axs.text(j, i, np.round(dtheta.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    
    ax.title.set_text(r'$\text{max}(\theta_i) - \text{min}(\theta_i)$')
    #ax.set_xlabel(r'$r_0$')
    #plt.xlim([0.1, 2.10])
    
    ax.set_xticks(ticks= np.arange(0, r0_vals.size, step = 2), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::2]])
    #plt.ylim([0.1, 2.10])
    #ax.set_ylabel(r'$\delta$')
    ax.set_yticks(ticks= np.arange(0, delta_vals.size, step = 2), 
               labels=[fr'${delta:.1f}$' for delta in delta_vals[::2]])
    fig.colorbar(im, ax=ax)


    ax = axs[1, 0]
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = ax.imshow(dspeed.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # for i, r0 in enumerate(r0_vals):
    #     for j, delta in enumerate(delta_vals):
    #         if not np.isnan(drtheta.T[i, j]):
    #             text = axs.text(j, i, np.round(drtheta.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    
    ax.title.set_text(r'$|\langle \omega_i \rangle - \omega_{r_0}|$')
    #ax.set_xlabel(r'$r_0$')
    #plt.xlim([0.1, 2.10])
    
    ax.set_xticks(ticks= np.arange(0, r0_vals.size, step = 2), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::2]])
    #plt.ylim([0.1, 2.10])
    #ax.set_ylabel(r'$\delta$')
    ax.set_yticks(ticks= np.arange(0, delta_vals.size, step = 2), 
               labels=[fr'${delta:.1f}$' for delta in delta_vals[::2]])
    fig.colorbar(im, ax=ax)


    ax = axs[1, 1]
    cmap = mpl.colormaps.get_cmap('inferno')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    im = ax.imshow(epsilons.T, 
               interpolation='none',
               cmap=cmap,
               origin='lower',
               resample=False)
    # for i, r0 in enumerate(r0_vals):
    #     for j, delta in enumerate(delta_vals):
    #         if not np.isnan(epsilons.T[i, j]):
    #             text = axs.text(j, i, np.round(epsilons.T[i, j], 2),
    #                    ha="center", va="center", color="w", size=10)
    
    ax.title.set_text(r'$\epsilon$')
    
    #plt.xlim([0.1, 2.10])
    
    ax.set_xticks(ticks= np.arange(0, r0_vals.size, step = 2), 
               labels=[fr'${r0:.1f}$' for r0 in r0_vals[::2]])
    #plt.ylim([0.1, 2.10])
    #ax.set_ylabel(r'$\delta$')
    ax.set_yticks(ticks= np.arange(0, delta_vals.size, step = 2), 
               labels=[fr'${delta:.1f}$' for delta in delta_vals[::2]])
    fig.colorbar(im, ax=ax)

"""