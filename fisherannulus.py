# Fisher on an Annulus
#fisher-refactored
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import gc
import simutils
import os
import datetime
import pickle

plt.rcParams['font.size'] = '32'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


def full_simulation(T, num_steps, dt, mesh_path, raw_bcs, u_0, \
                    path_and_simid, model_params, mode, output_ims, output_soln):
    
    dbar = float(model_params[0]['dbar'])
    sbar = float(model_params[0]['sbar'])
    
    themesh, V, Q, u_0, a, L, u, u_n1, u_n2, bcs, ds, boundary_markers = simutils.setup_fisher(mesh_path, 
                                        raw_bcs, u_0, dt, sbar, dbar)


    # Time-stepping
    t = 0
    if mode == 'isoprops':

        fig, (ax1, speed_ax, ang_speed_ax) = plt.subplots(3, figsize=(8,8))
        all_angles = np.array([])
        angle_means = np.array([])
        angle_max = np.array([])
        angle_min = np.array([])
        all_kappas = np.array([])
        all_r = np.array([])
        r_means = np.array([])
        kappa_means = np.array([])
        times_1 = np.array([])
        times_2 = np.array([])
        all_speeds = np.array([])
        speed_means = np.array([])
        speed_maxs = np.array([])
        speed_mins = np.array([])
        all_angular_speeds = np.array([])
        angular_speed_means = np.array([])
        all_x = np.array([])
        all_y = np.array([])
        mean_x = np.array([])
        mean_y = np.array([])

    elif mode == 'contour':
        fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [9, 1]}, figsize=(10,10))
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
            x_vals, y_vals, iso_idxs = simutils.get_indices_isoline(u, themesh, Q, iso=iso, annulus=True)
            r_vals = np.power(np.power(x_vals, 2) + np.power(y_vals, 2), 0.5)
            if np.array_equal(x_vals,y_vals) and y_vals.size == 0:
                pass
            
            else:
                angles, kappa = simutils.calculate_isoline_properties_annulus(u, themesh, Q, iso_idxs)
                speed = simutils.calculate_speed_fisher(u, themesh, Q, iso_idxs, model_params)
                # Calculate angular speed to check how it varies as r changes
                angular_speeds_direct = speed / r_vals
                
                # Means
                angle_mean = np.mean(angles)
                if angle_mean < -1.5:
                    break
                angle_means = np.append(angle_means, np.mean(angles))
                angle_max = np.append(angle_max, np.max(angles))
                angle_min = np.append(angle_min, np.min(angles))
                kappa_means = np.append(kappa_means, np.mean(kappa))
                all_angles = np.append(all_angles, angles)
                times_1 = np.append(times_1, t * np.ones(len(angles)))
                all_kappas = np.append(all_kappas, kappa)
                times_2 = np.append(times_2, t)

                speed_means = np.append(speed_means, np.mean(speed))
                speed_maxs = np.append(speed_maxs, np.max(speed))
                speed_mins = np.append(speed_mins, np.min(speed))
                all_speeds = np.append(all_speeds, speed)
                r_means = np.append(r_means, np.mean(r_vals))
                all_r = np.append(all_r, r_vals)

                all_x = np.append(all_x, x_vals)
                all_y = np.append(all_y, y_vals)
                mean_x = np.append(mean_x, np.mean(x_vals))
                mean_y = np.append(mean_y, np.mean(y_vals))

                angular_speed_means = np.append(angular_speed_means, 
                                                np.mean(angular_speeds_direct))
                all_angular_speeds = np.append(all_angular_speeds, angular_speeds_direct)



                if output_soln:
                    simutils.save_soln([x_vals, y_vals, angles, kappa, speed], os.path.join(path_and_simid, 'soln'), str(np.round(t, decimals=4)))

        elif mode == 'contour':
            x_vals, y_vals, u_vals = simutils.extract_coordinates(u, themesh, V, annulus=True)
            
            x_vals_iso, y_vals_iso, iso_idxs = simutils.get_indices_isoline(u, themesh, Q, iso=iso, annulus=True)
            r_vals_iso = np.power(np.power(x_vals_iso, 2) + np.power(y_vals_iso, 2), 0.5)
            sorted_iso_idxs = np.argsort(r_vals_iso)
            simutils.arrange_contour_plot(x_vals, y_vals, u_vals, axs[0], axs[1], 'Solution at t={0}'.format(round(t, 4)), 
                                          plot_border=True, vmax=1, paper=True)
            axs[0].set_adjustable('box')

            axs[0].plot(x_vals_iso[sorted_iso_idxs], y_vals_iso[sorted_iso_idxs], color='black', lw=4)
            if output_soln:
                simutils.save_soln([x_vals, y_vals, u_vals], os.path.join(path_and_simid, 'soln'), '{:.2f}'.format(np.round(t, decimals=4)))
            
            
            if output_ims:
                simutils.save_img(os.path.join(path_and_simid, 'imgs'), str(np.round(t, decimals=4)))




        assign(u_n2, u_n1)
        assign(u_n1, u)
        print('Step {0}.'.format(n))
        gc.collect()


        if mode == 'contour':
            axs[0].clear()
            axs[1].clear()


    if mode == 'isoprops':
        theoretic_mean = 2 * (sbar * dbar) ** 0.5
        angleplot1, = ax1.plot(times_1, np.pi/2 - all_angles, '.', label='Individual samples')
        angleplot2, = ax1.plot(times_2, np.pi/2 - angle_means, 'r', label='Mean angle')
        posplot1, = ax1.plot(times_2, np.pi/2 - angle_max, color='orange')
        posplot3, = ax1.plot(times_2, np.pi/2 - angle_min, color='magenta')
        posplot4 = ax1.fill_between(times_2, np.pi/2 - np.array(angle_max), np.pi/2 - np.array(angle_min), color='grey')
        
        ax1.set_ylabel('Distance (annulus midline)')
        ax1.legend(handles=[angleplot1, angleplot2])

        fig.suptitle('Time evolution of the u={0} isoline.'.format(iso))

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        plt.tight_layout()
        plt.savefig(path_and_simid + '/-' + timestamp + '.png', format='png')
    
        modulator = simutils.color_assign_criteria(all_r, Re, Ri)


        speed_fig, speed_ax = plt.subplots()
        speed_ax.scatter(times_1, all_speeds, marker='o', c=modulator)
        speed_ax.plot(times_2, speed_maxs, c='orange')
        speed_ax.plot(times_2, speed_mins, c='magenta')
        speed_ax.plot(times_2, speed_means, c='red')

        theoretic_mean_speed_plot, = speed_ax.plot(times_2, theoretic_mean * np.ones((times_2.size)),
                                                label='Theoretical straight domain')

        speed_ax.set_xlabel('Time')
        speed_ax.set_ylabel('Speed (non-dim)')
        plt.tight_layout()
        plt.savefig(path_and_simid + '/-speedfig' + timestamp + '.png', format='png')
        
        ang_speed_ax.scatter(times_1, all_angular_speeds, marker='o', c=modulator)
        ang_speed_ax.plot(times_2, angular_speed_means, c='orange')
        ang_speed_ax.set_xlabel('Time')
        ang_speed_ax.set_ylabel('Angular speed')
        plt.tight_layout()
        plt.savefig(path_and_simid + '/-ang_speedfig' + timestamp + '.png', format='png')
        

        results_dict = {'times_1': times_1, 'times_2': times_2,
                        'all_angles': all_angles, 'angle_means': angle_means,
                        'angle_max': angle_max, 'angle_min': angle_min,
                        'all_kappas': all_kappas, 'all_r': all_r,
                        'r_means': r_means, 'kappa_means': kappa_means,
                        'all_speeds': all_speeds, 'speed_means': speed_means,
                        'speed_maxs': speed_maxs, 'speed_mins': speed_mins,
                        'all_angular_speeds': all_angular_speeds,
                        'angular_speed_means': angular_speed_means,
                        'all_x': all_x, 'all_y': all_y,
                        'mean_x': mean_x, 'mean_y': mean_y

        }
        return results_dict

    else:
        return

# Define the fisher simulation and set up parameters
simfile = 'baseline-halfannulus-fine.json'
savepath = 'solns/fisher/'
mode = 'isoprops' # 'isoprops' or 'contour'

iso = 0.5
output_ims = True
output_soln = True

T, num_steps, dt, Re, Ri, tol, model_params, mesh_path, raw_bcs, u_0, simid = simutils.load_curved_fisher(simfile)

r0 = (Re + Ri)/2

path_and_simid = os.path.join(savepath, simid)
simutils.prepare_folders(savepath, simid)

results = full_simulation(T, num_steps, dt, mesh_path, raw_bcs, u_0, \
                    path_and_simid, model_params, mode, output_ims, output_soln)
