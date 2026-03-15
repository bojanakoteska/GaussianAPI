# Compare the prediction quality of different gaussian processes or skf splines on a certain
# validation set.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset

class Prediction_Stats:
    def __init__(self, validfile):
        self.validfile = validfile
        self.models = {} # results dictionary
        self.Comp_val = None # force components of validfile
        self.Mod_val = [] # moduli of force vectors of validfile

    def add_model(self, mod, name):

        if self.Comp_val is None:
            self.Comp_val, self.B_val, self.R_val, self.D_val = mod.read_training_data_2bF(self.validfile)
            # calculate the moduli of the force vectors of validfile
            for i in range(len(self.R_val)):
                j = 3*i
                fx = self.Comp_val[j]
                fy = self.Comp_val[j+1]
                fz = self.Comp_val[j+2]
                self.Mod_val += [np.sqrt(fx**2+fy**2+fz**2)]
            self.Mod_val = np.array(self.Mod_val)

        # prediction
        Comp_pred = mod.system_2bF_predict(self.B_val, self.R_val, self.D_val)
        Mod_pred = []

        # calculate the moduli of the predicted force vectors
        for i in range(len(self.R_val)):
            j = 3*i
            fx = Comp_pred[j]
            fy = Comp_pred[j+1]
            fz = Comp_pred[j+2]
            Mod_pred += [np.sqrt(fx**2+fy**2+fz**2)]
        Mod_pred = np.array(Mod_pred)

        # calculate residuals
        Comp_res = self.Comp_val - Comp_pred
        Mod_res = self.Mod_val - Mod_pred

        # calculate the moduli of the residual vectors
        Vec_res = []
        for i in range(len(self.R_val)):
            j = 3*i
            fx = Comp_res[j]
            fy = Comp_res[j+1]
            fz = Comp_res[j+2]
            Vec_res += [np.sqrt(fx**2+fy**2+fz**2)]
        Vec_res = np.array(Vec_res)

        # calculate relative residuals
        Comp_res_rel = Comp_res/self.Comp_val
        Mod_res_rel = Mod_res/self.Mod_val

        # calculate root mean square errors of force components and moduli
        RMSE_comp = np.sqrt(np.sum(Comp_res**2)/len(Comp_res))
        RMSE_comp_rel = np.sqrt(np.sum(Comp_res_rel**2)/len(Comp_res_rel))
        RMSE_mod = np.sqrt(np.sum(Mod_res**2)/len(Mod_res))
        RMSE_mod_rel = np.sqrt(np.sum(Mod_res_rel**2)/len(Mod_res_rel))

        # calculate mean absolute errors of force components and moduli
        MAE_comp = np.sum(np.absolute(Comp_res))/len(Comp_res)
        MAE_comp_rel = np.sum(np.absolute(Comp_res_rel))/len(Comp_res_rel)
        MAE_mod = np.sum(np.absolute(Mod_res))/len(Mod_res)
        MAE_mod_rel = np.sum(np.absolute(Mod_res_rel))/len(Mod_res_rel)

        self.models[name] = {'Vec_res': Vec_res,
                             'Comp_pred':Comp_pred,
                             'Comp_res_rel':Comp_res_rel,
                             'Mod_pred':Mod_pred,
                             'Mod_res_rel':Mod_res_rel,
                             'RMSE_comp':RMSE_comp,
                             'RMSE_comp_rel':RMSE_comp_rel,
                             'RMSE_mod':RMSE_mod,
                             'RMSE_mod_rel':RMSE_mod_rel,
                             'MAE_comp':MAE_mod,
                             'MAE_comp_rel':MAE_comp_rel,
                             'MAE_mod':MAE_mod,
                             'MAE_mod_rel':MAE_mod_rel}

    def hist_Vec_res(self, data, binwidth=1, xlims=None, ylims=None,
                     relative=False, alpha=0.5, weighted=False,
                     figsize=None, bin_boundaries=None,
                     ip_list=[0.4, 0.2, 0.5, 0.5],
                     plot_inset=False,
                     F_min=0.,
                     filename='hist_Vec_res.pdf'):
        # data = [(name, label, color)]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_ylabel(r'$\mathrm{frequency} \ (\%)$', fontsize=20)
        ax.set_xlabel(r'$ \mathrm{|residual \ vector|} \ (\mathrm{eV/ \AA})$',
                      fontsize=20)
        if plot_inset:
            ax2 = plt.axes([0,0,1,1])
            ip = InsetPosition(ax, ip_list)
            ax2.set_axes_locator(ip)
            loc1 = 3
            loc2 = 1
            mark_inset(ax, ax2, loc1=loc1, loc2=loc2, fc="none", ec='0.5')
        for tup in data:
            y = self.models[tup[0]]['Vec_res']
            label = tup[1]
            color = tup[2]
            num_bins = int(round(np.amax(y)/binwidth))
            if bin_boundaries is None:
                bin_boundaries = np.linspace(0., np.amax(np.amax(y)), num_bins + 1)

            weights = np.ones(len(y))
            if weighted:
                weights = 100*weights/len(y)
            n, bins, patches = ax.hist(y, bins=bin_boundaries, facecolor=color, 
                                       normed=0, alpha=alpha, edgecolor='Black',
                                       label=label, weights=weights)
            if plot_inset:
                n2, bins2, patches2 = ax2.hist(y[y>=F_min],
                                    bins=bin_boundaries[bin_boundaries>=F_min],
                                               facecolor=color, normed=0,
                                               alpha=alpha, edgecolor='Black',
                                               label=label,
                                               weights=weights[y>=F_min])

        ax.legend(loc='best', fontsize=15)
        if xlims is not None:
            ax.set_xlim((xlims[0],xlims[1]))
        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])
        ax.grid(True)
        ax.tick_params(labelsize=15)
        if plot_inset:
            ax2.grid(True)
            ax2.tick_params(labelsize=12)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

        return n, bins, patches

    def ref_vs_pred(self, data, xlims=None, ylims=None, alpha=1., figsize=(10,5),
                    mode='components', relative=False, wklhalb_scale=1.2,
                    filename='ref_vs_pred.pdf'):
        # data = [(name, label, color)]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        if not relative:
            ax.set_ylabel(r'$\mathrm{|predicted \ force|} \ \mathrm{(eV/ \AA)}$', fontsize=20)
            ax.set_xlabel(r'$ \mathrm{|reference \ force|} \ \mathrm{(eV/ \AA)}$', fontsize=20)
            for tup in data:
                if mode == 'components':
                    prediction = self.models[tup[0]]['Comp_pred']
                    reference = self.Comp_val
                elif mode == 'moduli':
                    prediction = self.models[tup[0]]['Mod_pred']
                    reference = self.Mod_val

                label = tup[1]
                color = tup[2]
                # scatterplot
                ax.plot(reference, prediction, color=color, 
                        linewidth=0., marker='o', markeredgecolor='black', 
                        markersize=5, label=label, alpha=alpha)

            if mode == 'components':
                wklhalb_min = wklhalb_scale * np.amin(self.Comp_val)
                wklhalb_max = wklhalb_scale * np.amax(self.Comp_val)
            elif mode == 'moduli':
                wklhalb_min = 0.
                wklhalb_max = wklhalb_scale * np.amax(self.Comp_val)
            wklhalb = np.linspace(wklhalb_min, wklhalb_max)
            ax.plot(wklhalb, wklhalb, linestyle='--', color='Black', linewidth=2.)

        elif relative:
            ax.set_ylabel(r'$\mathrm{relative \ prediction \ error}$', fontsize=20)
            ax.set_xlabel(r'$ \mathrm{reference} \ \mathrm{(eV/ \AA)}$', fontsize=20)
            for tup in data:
                if mode == 'components':
                    prediction = self.models[tup[0]]['Comp_res_rel']
                    reference = self.Comp_val
                elif mode == 'moduli':
                    prediction = self.models[tup[0]]['Mod_res_rel']
                    reference = self.Mod_val

                label = tup[1]
                color = tup[2]
                # scatterplot
                ax.plot(reference, prediction, color=color, 
                        linewidth=0., marker='o', markeredgecolor='black', 
                        markersize=5, label=label, alpha=alpha)

        ax.legend(loc='best', fontsize=15)
        if xlims is not None:
            ax.set_xlim((xlims[0],xlims[1]))
        elif not relative:
            ax.set_xlim(wklhalb[0], wklhalb[-1])
        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])
        elif not relative:
            ax.set_ylim(wklhalb[0], wklhalb[-1])
        ax.grid(True)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()








