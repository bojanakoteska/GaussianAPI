import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset
from ase.units import Bohr, Hartree

def plot_repulsion(X, bond_type, mod_dict = {},
                   xlims=None, ylims=None,
                   title=None,
                   R_min = 1.5,
                   der=0, figsize=(10,5),
                   legend_cols=1, ip_list=[0.4,0.2,0.5,0.5],
                   filename='repulsion_plot.pdf',
                   fs1 = 20,
                   lfs = 15,
                   ls1 = 15,
                   ls2 = 12):
    # mod_dict = {'name':{'type':'GP'/'Spline',
    #                     'model':modelobject,
    #                     'label':label,
    #                     'color':color,
    #                     'Rmin':Rmin
    #                     'linestyle':e.g. '--'}}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.grid(True)
    if title is not None:
        ax.set_title(title, fontsize=fs1)
    ax.set_xlabel(r'$R \ (\mathrm{\AA})$', fontsize=fs1)

    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax, ip_list)
    ax2.set_axes_locator(ip)
    if der == 0:
        loc1 = 3
        loc2 = 1 
    else:
        loc1 = 2
        loc2 = 4
    mark_inset(ax, ax2, loc1=loc1, loc2=loc2, fc="none", ec='0.5')

    if der == 0:
        ax.set_ylabel(r'$V_{\mathrm{rep}} \ (\mathrm{eV})$', fontsize=fs1)
    if der == 1:
        ax.set_ylabel(r'$F_{\mathrm{rep}} \ (\mathrm{eV/ \AA})$', fontsize=fs1)
    for name in mod_dict.keys():
        if mod_dict[name]['type'] == 'GP':
            if 'linestyle' in mod_dict[name].keys():
                linestyle = mod_dict[name]['linestyle']
            else:
                linestyle='-'
            if der == 0:
                Y = - mod_dict[name]['model'].pot_vpredict(X, bond_type)
            elif der == 1:
                Y = - mod_dict[name]['model'].vpredict(X, bond_type=bond_type, der=1)
        elif mod_dict[name]['type'] == 'Spline':
            if 'linestyle' in mod_dict[name].keys():
                linestyle = mod_dict[name]['linestyle']
            else:
                linestyle='--'
            if der == 0:
                Y = mod_dict[name]['model'].vpredict(X, bond_type=bond_type, der=0)
            elif der == 1:
                Y = mod_dict[name]['model'].vpredict(X, bond_type=bond_type, der=1)
        label = mod_dict[name]['label']
        color = mod_dict[name]['color']
        if 'X_min' in mod_dict[name].keys():
            X_min = mod_dict[name]['X_min']
        else:
            X_min = X[0]
        if 'X_max' in mod_dict[name].keys():
            X_max = mod_dict[name]['X_max']
        else:
            X_max = X[-1]

        ax.plot(X[(X>=X_min) & (X<=X_max)], Y[(X>=X_min) & (X<=X_max)],
                linestyle=linestyle, linewidth=2., color=color, label=label)
        ax2.plot(X[(X>R_min) & (X<=X_max)], Y[(X>R_min) & (X<=X_max)],
                 linestyle=linestyle, linewidth=2., color=color)
                
    ax2.grid(True)
    if xlims is not None:
        ax.set_xlim(xlims[0],xlims[1])
    if ylims is not None:
        ax.set_ylim(ylims[0],ylims[1])
    ax.legend(loc='best', ncol=legend_cols, fontsize=lfs)

    ax.tick_params(labelsize=ls1)
    ax2.tick_params(labelsize=ls2)

    plt.savefig(filename,
                bbox_inches = 'tight')
    plt.show()
    return

def plot_RMSE_vs_hyper(hyper, RMSE, xlims=None, ylims=None, figsize=None,
                       color='CornflowerBlue', xlabel='', title='', log=False,
                       filename='RMSE_vs_hyper.pdf'):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(r'$\mathrm{RMSE} \ (\mathrm{eV/\AA})$', fontsize=20)
    ax.set_title(title, fontsize=20)
    if not log:
        ax.plot(hyper, RMSE, marker='o', color=color, markersize=5, linewidth=2.)
    else:
        ax.semilogx(hyper, RMSE, marker='o', color=color, markersize=5, linewidth=2.)
    ax.grid(True)
    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    return

def visualize_training_data(bond_file, bond_type,
                            figsize=None,
                            title=None,
                            R_max=None,
                            num_bins=100,
                            filename='bond_histo.pdf'):
    bond_type2 = bond_type.split('-')[1] + '-' + bond_type.split('-')[0]
    with open(bond_file, 'r') as bf:
        bond_dists = []
        for line in bf:
            if line == bond_type + '\n' or line == bond_type2 + '\n':
                break
        for line in bf:
            if line.startswith('\n'):
                break
            bond_dists += [float(line)]
        bond_dists = np.array(bond_dists)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.grid(True)
        if title is not None:
            ax.set_title(title, fontsize=20)
        ax.set_xlabel(r'$R \ (\mathrm{\AA})$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{number \ of \ bonds}$', fontsize=20)
        if R_max is not None:
            bond_dists = bond_dists[bond_dists<R_max]
        ax.hist(bond_dists, num_bins, normed=0, facecolor='CornflowerBlue',alpha=0.75,
                edgecolor='Black', label=r'$\mathrm{bond \ distances}$')

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        
def histo_force_components(components,
                           figsize=None,
                           title=None,
                           filename='comp_histo.pdf',
                           weighted=True,
                           binwidth=1.,
                           bin_boundaries=None,
                           ip_list=[0.4, 0.2, 0.5, 0.5],
                           plot_inset=False,
                           F_min=0.,
                           color='CornflowerBlue',
                           alpha=0.75,
                           ls1=15,
                           ls2=12):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.grid(True)
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel(r'$|F_{i}| \ (\mathrm{eV/\AA})$', fontsize=20)
    ax.set_ylabel(r'$\mathrm{frequency} \ (\%)$', fontsize=20)
    if plot_inset:
        ax2 = plt.axes([0,0,1,1])
        ip = InsetPosition(ax, ip_list)
        ax2.set_axes_locator(ip)
        loc1 = 3
        loc2 = 1
        mark_inset(ax, ax2, loc1=loc1, loc2=loc2, fc="none", ec='0.5')
    
    num_bins = int(round(np.amax(components)/binwidth))
    if bin_boundaries is None:
        bin_boundaries = np.linspace(0, np.amax(components), num_bins + 1,
                                     dtype=int)
    weights = np.ones(len(components))
    if weighted:
        weights = 100*weights/len(components)
    ax.hist(components,
            bins=bin_boundaries,
            normed=0,
            facecolor=color,
            alpha=alpha,
            edgecolor='Black', weights=weights)
    if plot_inset:
        ax2.hist(components[components>=F_min],
                 bins=bin_boundaries[bin_boundaries>=F_min],
                 facecolor=color, normed=0,
                 alpha=alpha, edgecolor='Black',
                 weights=weights[components>=F_min])

    ax.tick_params(labelsize=ls1)
    if plot_inset:
        ax2.grid(True)
        ax2.tick_params(labelsize=ls2)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_theta_sigma_RMSE(theta_values, sigma_values, RMSE, num_lines=20, cmap='RdGy',
                          figsize=None, log_y=True, log_x=False,
                          filename='theta_sigma_RMSE.pdf',
                          min_grdpnt=[0,0],NoShow=False):
    RMSE = np.reshape(np.array(RMSE), (len(theta_values), len(sigma_values))).T
    Theta, Sigma = np.meshgrid(theta_values, sigma_values)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\theta \ (\mathrm{\AA})$', fontsize=15)
    ax.set_ylabel(r'$\sigma_{n} \ (eV)$', fontsize=15)


    cax = ax.contourf(Theta, Sigma, RMSE, num_lines, cmap=cmap)
    cbar = fig.colorbar(cax)
    cbar.set_label(r'$\mathrm{RMSE}  \ (\mathrm{eV} / \mathrm{\AA})$', fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    if log_y:
        ax.set_yscale('log')

    if log_x:
        ax.set_xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.plot(Theta, Sigma, marker='o', markersize=5, linewidth=0., color='LightSkyBlue',
            alpha=0.2)
    ax.plot(min_grdpnt[0], min_grdpnt[1], marker='o', markersize=5, linewidth=0., color='Black')

    plt.savefig(filename, bbox_inches='tight')

    if not NoShow:
        plt.show()


def plot_theta_theta_RMSE(theta_values1, theta_values2, RMSE, num_lines=20, cmap='RdGy',
                          figsize=None, log_y=False, log_x=False,
                          xlabel=r'$\theta_{1}$',
                          ylabel=r'$\theta_{2}$',
                          filename='theta_theta_RMSE.pdf',
                          min_grdpnt=[0,0]):
    RMSE = np.reshape(np.array(RMSE), (len(theta_values1), len(theta_values2))).T
    Theta1, Theta2 = np.meshgrid(theta_values1, theta_values2)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)


    cax = ax.contourf(Theta1, Theta2, RMSE, num_lines, cmap=cmap)
    cbar = fig.colorbar(cax)
    cbar.set_label(r'$\mathrm{RMSE}  \ (\mathrm{eV} / \mathrm{\AA})$', fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    if log_y:
        ax.set_yscale('log')

    if log_x:
        ax.set_xscale('log')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.plot(Theta1, Theta2, marker='o', markersize=5, linewidth=0., color='LightSkyBlue',
            alpha=0.2)
    ax.plot(min_grdpnt[0], min_grdpnt[1], marker='o', markersize=5, linewidth=0., color='Black')

    plt.savefig(filename, bbox_inches='tight')

    plt.show()
