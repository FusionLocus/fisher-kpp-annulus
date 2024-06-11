# Fisher on a rectangle
#fisher-refactored
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import gc
import simutils
import os
import datetime

plt.rcParams['font.size'] = '32'
plt.rcParams['text.usetex'] = True

def full_simulation(T, num_steps, dt, mesh_path, raw_bcs, u_0, \
                    path_and_simid, model_params, mode, output_ims, output_soln): 
    sbar = float(model_params[0]['sbar'])
    dbar = float(model_params[0]['dbar'])
    themesh, V, Q, u_0, a, L, u, u_n1, u_n2, bcs, ds, boundary_markers = simutils.setup_fisher(mesh_path, 
                                                raw_bcs, u_0, dt, sbar, dbar)

    # Time-stepping
    t = 0

    if mode == 'isoprops':

        fig, (ax1, speed_ax) = plt.subplots(2)
        all_x = np.array([])
        x_means = []
        x_max = []
        x_min = []
        all_y = np.array([])
        
        times_1 = np.array([])
        times_2 = np.array([])

        all_speeds = np.array([])
        speed_means = np.array([])
        speed_maxs = np.array([])
        speed_mins = np.array([])

    elif mode == 'contour':
        fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [9, 1]}, figsize=(10,2))
        axs[0].set_aspect(1)

    for n in range(num_steps):
        # Update current time
        maxu = project(u, V).vector().max()
        minu = project(u, V).vector().min()
        print('At t={0}, max={1}, min={2}'.format(t, maxu, minu))
        t += dt
        
        # Compute solution
        if not bcs:
            solve(a == L, u)
        elif len(bcs) == 1:
            solve(a == L, u, bcs[0])
        else:
            solve(a == L, u, bcs)

        if mode == 'isoprops':
            # Get isolines and extract their properties.
            x_vals, y_vals, iso_idxs = simutils.get_indices_isoline(u, themesh, Q, annulus=False)
            #kappa = utilities.calculate_isoline_properties_rect(u, themesh, Q, iso_idxs)
            if np.array_equal(x_vals,y_vals) and y_vals.size == 0:
                pass
            else:

                speed = simutils.calculate_speed_fisher(u, themesh, Q, iso_idxs, model_params, annulus=False)
                
                # Means
                x_means.append(np.mean(x_vals))
                x_max.append(np.max(x_vals))
                x_min.append(np.min(x_vals))

                all_x = np.append(all_x, x_vals)
                times_1 = np.append(times_1, t * np.ones(len(x_vals)))

                times_2 = np.append(times_2, t)

                speed_means = np.append(speed_means, np.mean(speed))
                speed_maxs = np.append(speed_maxs, np.max(speed))
                speed_mins = np.append(speed_mins, np.min(speed))
                all_speeds = np.append(all_speeds, speed)
                all_y = np.append(all_y, y_vals)

        elif mode == 'contour':
            x_vals, y_vals, u_vals = simutils.extract_coordinates(u, themesh, V, annulus=True)
            simutils.arrange_contour_plot(x_vals, y_vals, u_vals, axs[0], axs[1], 
                                        'Cell density at t={0}'.format(round(t, 4)),
                                        vmax=1, paper=True, annulus=False, plot_border=True)
            axs[0].set_adjustable('box')
            
            x_vals_iso, y_vals_iso, iso_idxs = simutils.get_indices_isoline(u, themesh, Q, iso=iso, annulus=False)
            sorted_iso_idxs = np.argsort(y_vals_iso)
            axs[0].plot(x_vals_iso[sorted_iso_idxs], y_vals_iso[sorted_iso_idxs], color='black', lw=4)


            if output_soln:
                simutils.save_soln([x_vals, y_vals, u_vals], os.path.join(path_and_simid, 'soln'), str(np.round(t, decimals=4)))
            
            if output_ims:
                simutils.save_img(os.path.join(path_and_simid, 'imgs'), str(np.round(t, decimals=4)))

        # Assign this solution to the previous timestep
        assign(u_n1, u)
        print('Step {0}.'.format(n))
        gc.collect()

        if mode == 'contour':
            axs[0].clear()
            axs[1].clear()


    if mode == 'isoprops':
        posplot1, = ax1.plot(times_2, x_max, color='orange')
        posplot3, = ax1.plot(times_2, x_min, color='orange')
        posplot4 = ax1.fill_between(times_2, np.array(x_max), np.array(x_min), color='grey')
        posplot2, = ax1.plot(times_2, x_means, color='red', label='Mean position')
        ax1.set_ylabel('x position')
        ax1.legend(handles=[posplot2])

        theoretic_mean = 2 * (sbar * dbar) ** 0.5
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        modulator = simutils.color_assign_criteria(all_y, 0, Ly)


        speed_ax.scatter(times_1, all_speeds, marker='o', c=modulator)
        speed_ax.plot(times_2, speed_means, c='red')
        speed_ax.plot(times_2, theoretic_mean * np.ones((times_2.size)), c='black', label='Infinite line domain')
        speed_ax.set_xlabel('Time')
        speed_ax.set_ylabel('Speed (non-dim)')

        plt.savefig(path_and_simid + '/-speedfig' + timestamp + '.png', format='png')
        
        results_dict = {'times_1': times_1, 'times_2': times_2,
                        'all_x': all_x, 'all_y': all_y,
                        'all_speeds': all_speeds, 'speed_means': speed_means,
                        'speed_maxs': speed_maxs, 'speed_mins': speed_mins
        }
        return results_dict

    else:
        return


# Define the fisher simulation and set up parameters
bulk_param = ''
bulk_param_vals = []
simfile = 'fisher-sim-parameters/rect-fine.json'
mode = 'isoprops' # '3D' or 'isoprops'
savepath = 'solns/'


iso = 0.5
output_ims = True
output_soln = True

T, num_steps, dt, Lx, Ly, tol, model_params, mesh_path, raw_bcs, u_0, simid = simutils.load_rect_fisher(simfile)

path_and_simid = os.path.join(savepath, simid)
simutils.prepare_folders(savepath, simid)

results = full_simulation(T, num_steps, dt, mesh_path, raw_bcs, u_0, \
                path_and_simid, model_params, mode, output_ims, output_soln)


