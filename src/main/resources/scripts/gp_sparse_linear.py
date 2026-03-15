import numpy as np
import itertools
import pickle as cPickle
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares, curve_fit
from scipy.integrate import quad
import covariance
from copy import copy
from ase.units import Bohr, Hartree

class GP_sparse_linear:
    """
    Original code written by Artur Engelmann. 
    Additional functionality and maintenance by Johannes Margraf
    """

###############################################################################
# I. Initialization                                                           #
###############################################################################

    def __init__(self,
                 training_file='training.dat',
                 sigma = 1.,
                 bond_dict={},
                 kernel='SE',
                 eps_K_M=10.**-6,
                 auto_train=True,
                 cholesky_Q_M=True,
                 nbody=2,
                 rel_sigma=False,
                 submean=False):
        """
        initialization and precomputation
        """

        # extra term (small) on diagonal of sparse covariance matrix for
        # numerical stability
        self.eps_K_M = eps_K_M

        # inversion of Q_M via cholesky decompostion or directly?
        self.cholesky_Q_M = cholesky_Q_M

        # vectorizations
        self.pot_vpredict = np.vectorize(self.pot_predict)

        # how many sets in cross validation? only change this variable
        # via the function self.gen_crossindcs(k_cross)!
        self.k_cross = None

        # type of n-body descriptor
        self.nbody = nbody

        # use relative error per force component
        self.rel_sigma = rel_sigma
        # subtract mean
        self.submean = submean
        self.ymean = 0.0

        # hyperparameters
        # ===============
        # the user has to specify all the participating bond types.
        # this is done by setting the value of the variable 'bond_dict', which
        # also contains the hyperparameters for every bond type.
        # e.g. for a two component system AB,
        # bond_dict = {'AA':{'delta':xxx, 'theta':xxx, 'beta':xxx,
        #                    'cutoff':xxx, 'd':xxx},
        #              'BB':{...}, 'AB':{...}}
        self.bond_dict = bond_dict
        self.bond_types = [] # list of all bond types (for later loops in
                             # matrix constructions we need them in order)

        # define (bad!) default values of unspecified hyperparameters
        for bond_type in self.bond_dict.keys():
            self.bond_types += [bond_type]
            if not 'delta' in self.bond_dict[bond_type].keys():
                self.bond_dict[bond_type]['delta'] = 1.
            if not 'theta' in self.bond_dict[bond_type].keys():
                self.bond_dict[bond_type]['theta'] = 1.
            if not 'beta' in self.bond_dict[bond_type].keys():
                self.bond_dict[bond_type]['beta'] = 1.
            if not 'cutoff' in self.bond_dict[bond_type].keys():
                self.bond_dict[bond_type]['cutoff'] = 1.
            if not 'd' in self.bond_dict[bond_type].keys():
                self.bond_dict[bond_type]['d'] = 1.

        # sigma is a global hyperparameter (noise of the training data)
        self.sigma = sigma

        # meaning of the hyperparameters:
            # delta: latent function standard deviation
            # theta: latent function length scale
            # beta: exponential damping factor
            # sigma: data noise standard deviation
            # cutoff: cutoff
            # d: cutoff transition width

        # pseudo input
        # ============
        # pseudo input data points need to specified for every bond type.
        # this is done in the the bond_dict
        for bond_type in self.bond_dict.keys():
            if not 'X_pseudo' in self.bond_dict[bond_type].keys():
                raise ValueError('You have to provide X_pseudo for the {}-bond!'.format(bond_type))
        # determine the lengths of the pseudo input arrays, 
        # self.M = total number of pseudo inputs
        self.M = 0
        for bond_type in self.bond_dict.keys():
            M =  len(self.bond_dict[bond_type]['X_pseudo'])
            self.bond_dict[bond_type]['M'] = M
            self.M += M

        # training data
        # =============
        if self.nbody == 2:
            self.Y, self.B, self.X, self.D = self.read_training_data_2bF(
                                                                     training_file)
            self.N = len(self.Y) # number of data points
        elif self.nbody == 1:
            self.Y, self.B, self.X         = self.read_training_data_1bE(
                                                                     training_file)
            self.N = len(self.Y) # number of data points

        if self.submean:
            self.ymean  = np.mean(self.Y)
            self.Y -= self.ymean 

        # covariance functions 
        # ====================
        self.kernel = kernel # kernel key
        # which covariance function to use (see IV)
        if self.kernel == 'SE':
            self.k_obj = covariance.K_SE()
            self.dx1_k_obj = covariance.dx1_K_SE()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_SE()
        elif self.kernel == 'SE_cut':
            self.k_obj = covariance.K_SE_cut()
            self.dx1_k_obj = covariance.dx1_K_SE_cut()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_SE_cut()
        if self.kernel == 'SE_damped':
            self.k_obj = covariance.K_SE_damped()
            self.dx1_k_obj = covariance.dx1_K_SE_damped()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_SE_damped()
        elif self.kernel == 'SE_damped_cut':
            self.k_obj = covariance.K_SE_damped_cut()
            self.dx1_k_obj = covariance.dx1_K_SE_damped_cut()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_SE_damped_cut()
        elif self.kernel == 'Lap':
            self.k_obj = covariance.K_Lap()
            self.dx1_k_obj = covariance.dx1_K_Lap()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_Lap()
        elif self.kernel == 'Lap_cut':
            self.k_obj = covariance.K_Lap_cut()
            self.dx1_k_obj = covariance.dx1_K_Lap_cut()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_Lap_cut()
        if self.kernel == 'RQ':
            self.k_obj = covariance.K_RQ()
            self.dx1_k_obj = covariance.dx1_K_RQ()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_RQ()
        elif self.kernel == 'RQ_cut':
            self.k_obj = covariance.K_RQ_cut()
            self.dx1_k_obj = covariance.dx1_K_RQ_cut()
            self.dx1_dx2_k_obj = covariance.dx1_dx2_K_RQ_cut()

        # training
        if auto_train:
            self.train()

###############################################################################
# II. Training and Prediction                                                 #
###############################################################################

    def train(self):
        """
        execute all linear algebra operations necessary to calculate alpha
        """
        self.calc_K_M()
        self.calc_K_NM()
        self.calc_Q_M()
        self.calc_alpha()

    def calc_K_M(self):
        """
        calculate K_M and its inverse inv_K_M
        """

        # calculate K_M
        self.K_M = covariance.calc_K_M(self.k_obj,
                                       self.bond_types,
                                       self.bond_dict,
                                       self.eps_K_M,
                                       self.M)

        #self.inv_K_M = np.linalg.inv(self.K_M)

        # calculate inverse of K_M via cholesky decomposition

        # cholesky decomposition of K_M
        #L_K_M = cholesky(self.K_M, lower=True)

        # calculate inverse of K_M
        #inv_L_K_M = solve_triangular(L_K_M, np.identity(self.M), lower=True)
        #self.inv_K_M = np.dot(inv_L_K_M.T, inv_L_K_M)


    def calc_K_NM(self):
        """
        calculate covariance matrix between training inputs and pseudo inputs
        K_NM and its transpose K_MN
        """
        if self.nbody == 2:
            self.K_NM = covariance.calc_K_NM_2bF(self.dx1_k_obj,
                                                 self.bond_types,
                                                 self.bond_dict,
                                                 self.N,
                                                 self.M,
                                                 self.B,
                                                 self.X,
                                                 self.D)
        elif self.nbody == 1:
            self.K_NM = covariance.calc_K_NM_1bF(self.k_obj,
                                                 self.bond_types,
                                                 self.bond_dict,
                                                 self.N,
                                                 self.M,
                                                 self.B,
                                                 self.X)
                                                 


        self.K_MN = self.K_NM.T


    def calc_Q_M(self):
        """
        calculate pseudo-covariance matrix of sparsified data set Q_M
        and its inverse inv_Q_M
        """
        # calculate Q_M
        if not self.rel_sigma:
            self.Q_M = self.K_M + self.sigma**-2 * np.dot(self.K_MN, self.K_NM)
        else:
            self.Q_M = self.K_M + self.sigma**-2 * np.dot(self.K_MN, self.K_NM)
        #    self.Q_M = self.K_M + self.sigma**-2*self.Y * np.dot(self.K_MN, self.K_NM)

        if not self.cholesky_Q_M:
            # calculate inverse directly
            self.inv_Q_M = np.linalg.inv(self.Q_M)
        else:
            # calculate inverse via cholesky decomposition

            # cholesky decomposition of Q_M
            L_Q_M = cholesky(self.Q_M, lower=True)

            # calculate inverse of Q_M
            inv_L_Q_M = solve_triangular(L_Q_M, np.identity(self.M), lower=True)
            self.inv_Q_M = np.dot(inv_L_Q_M.T, inv_L_Q_M)


    def calc_alpha(self):
        """
        calculate alpha (='expansion coefficients')
        """
        temp = np.dot(self.inv_Q_M, self.K_MN)
        self.alpha = self.sigma**-2 * np.dot(temp, self.Y)

    def vpredict(self, X_star, bond_type=None, der=0):
        if der == 0:
            k = self.k_obj
        elif der == 1:
            k = self.dx1_k_obj
        else:
            print ('der has to be 0 or 1!')
            return None

        if bond_type is None:
            bond_type = self.bond_types[0]

        return covariance.vpredict(k,
                                   bond_type,
                                   self.bond_types,
                                   self.bond_dict,
                                   self.M,
                                   self.alpha,
                                   X_star)

    def posterior_variance(self, X_star, bond_type=None, der=0):
        """
        calculate the the posterior function variance (minus the data noise) 
        """
        if der == 0:
            k = self.k_obj
        elif der == 1:
            k = self.dx1_k_obj
        else:
            print ('der has to be 0 or 1!')
            return None
        
        if bond_type is None:
            bond_type = self.bond_types[0]

        return covariance.posterior_variance(k,
                                             bond_type,
                                             self.bond_types,
                                             self.bond_dict,
                                             self.M,
                                             self.inv_K_M,
                                             self.inv_Q_M,
                                             X_star)





    # For some reason vpredict(X_star, der=1) is not reproduced by taking
    # the first derivative of vpredict(X_star, der=0). That the prediction
    # of the forces seems to be correct however, can be checked via the simple
    # symmetric structures.
    # Here we determine the potential by integrating the force numerically.

    def force_predict(self, x_star, bond_type=None):
        return self.vpredict(np.array([x_star]), bond_type, der=1)

    def pot_predict(self, x_star, bond_type=None):
        if bond_type is None:
            bond_type = self.bond_types[0]
        cutoff = self.bond_dict[bond_type]['cutoff']
        return (- quad(self.force_predict, x_star, cutoff,
                       args=(bond_type))[0])


    def system_2bF_predict(self, B_val, X_val, D_val):
        '''
        predict the total force components on the atoms of some structure:

        B_val[iR]: list of bond types of the bonds contributing to the iR-th
                   force (not component!), which we want to predict

        X_val[iR]: corresponding list of bond lengths

        D_val[i]: list of bond length gradients corresponding to the i-th force
                  component, which we want to predict

        when we iterate over force components, we access the corresponding
        bond lists via
        iR = (i-i%3)/3

        output: list of force components
        [F_1_x, F_1_y, F_1_z, ...,F_num_atoms_x, F_num_atoms_y, F_num_atoms_z]
        '''

        return covariance.system_2bF_predict(self.dx1_k_obj,
                                             self.bond_types,
                                             self.bond_dict,
                                             self.M,
                                             self.alpha,
                                             B_val,
                                             X_val,
                                             D_val) 

    def system_1bE_predict(self, B_val, X_val):
        '''
        predict the total energy of some structure:

        B_val[iR]: list of atom types  contributing to the energy
                   which we want to predict

        X_val[iR]: corresponding list of descriptors

        '''

        return covariance.system_1bE_predict(self.k_obj,
                                             self.bond_types,
                                             self.bond_dict,
                                             self.M,
                                             self.alpha,
                                             B_val,
                                             X_val) 


###############################################################################
# III. Validation                                                             #
###############################################################################

    def gen_crossindcs(self, k_cross):
        """
        prepare the index sets to
        divide the training data T into k_cross equally large random subsets T_i
        """
        self.k_cross = k_cross
        indcs = np.arange(len(self.X)) # total force (not component!) indices
        set_size = len(indcs)/k_cross
        np.random.shuffle(indcs)
        self.indx_sets = []
        for i in range(k_cross):
            if i == k_cross-1: # remainder is added to the last index set
                self.indx_sets += [ indcs[i*set_size:] ]
            else:
                self.indx_sets += [ indcs[i*set_size:(i+1)*set_size] ]


    def cross_validation(self,verbose=False):
        """
        perform the cross validation, i.e. train with T_i!=j and predict on T_j

        output: list of the self.k_cross residuals
        """

        if self.k_cross is None:
            print ('please run self.gen_crossindcs(k_cross) first')
            return 0

        # copy the original training data
        Y_org = copy(self.Y) 
        B_org = copy(self.B)
        X_org = copy(self.X)
        if self.nbody == 2:
            D_org = copy(self.D)

        # perform training and prediction
        residual_sets = []
        cross_values = []
        for i in range(self.k_cross):
            predict_indcs = self.indx_sets[i]
            train_indcs = np.array([], dtype=np.int64)
            for j in range(self.k_cross):
                if i == j:
                    continue
                train_indcs = np.append(train_indcs, self.indx_sets[j])
            
            # corresponding component indices
            predict_indcs_comp = np.array([], dtype=np.int64)
            for j in predict_indcs:
                predict_indcs_comp = np.append(predict_indcs_comp, 
                                               [3*j, 3*j+1, 3*j+2])
            train_indcs_comp = np.array([], dtype=np.int64)
            for j in train_indcs:
                train_indcs_comp = np.append(train_indcs_comp, 
                                             [3*j, 3*j+1, 3*j+2])

            # set the training data variables and train
            self.B = B_org[train_indcs]
            self.X = X_org[train_indcs]
            self.Y = Y_org[train_indcs]
            if self.nbody == 2:
                self.Y = Y_org[train_indcs_comp]
                self.D = D_org[train_indcs_comp]
            self.N = len(self.Y)

            self.train()

            # predict
            B_val = B_org[predict_indcs]
            X_val = X_org[predict_indcs]
            Y_val = Y_org[predict_indcs]
            if self.nbody == 2:
                Y_val = Y_org[predict_indcs_comp]
                D_val = D_org[predict_indcs_comp]
                residual = self.system_2bF_predict(B_val, X_val, D_val) - Y_val
                residual_sets += [residual]

            elif self.nbody == 1:
                for ipre in range(len(B_val)):
                    tmp = self.system_1bE_predict(B_val[ipre], X_val[ipre])
                    residual_sets.append(tmp - Y_val[ipre])
                    cross_values.append([Y_val[ipre],tmp])        
        # reset the training data
        self.Y = copy(Y_org)
        self.B = copy(B_org)
        self.X = copy(X_org)
        if self.nbody == 2:
            self.D = copy(D_org)
        self.N = len(self.Y)

        # calculate and return the mean RMSE of all predictions
        RMSE = 0.
        if self.nbody == 2:
            for residual in residual_sets:
                RMSE += np.sqrt(np.sum(residual**2)/len(residual))/self.k_cross
        elif self.nbody == 1:
            for ires,residual in enumerate(residual_sets):
                RMSE += residual**2/float(len(residual_sets))
                if verbose:
                    print (cross_values[ires][0], cross_values[ires][1])
            RMSE = np.sqrt(RMSE)
        return RMSE

    def direct_validation(self, validfile):
        '''
        predict on validfile and return RMSE
        '''
        Y_val, B_val, X_val, D_val = self.read_training_data_2bF(validfile)
        residual = self.system_2bF_predict(B_val, X_val, D_val) - Y_val
        RMSE = np.sqrt(np.sum(residual**2)/len(residual))

        return RMSE


    def hyper_sweep(self, hyperdict, validmode='cross', k_cross=None,
                    validfile=None, update=True):
        '''
        perform a hyperparameter sweep and calculate how the RMSE of the
        prediction on some validation set changes

        parameters
        ----------
        hyperdict: We pass the hyperparameters, which we wish to vary in the
                   following way:
                   hyperdict = {'theta_A-B':sweepvalues}.
                   We specify the name of the hyperparameter (e.g. 'theta') and
                   then after an underscore the corresponding bondtype
                   (e.g. 'A-B'). If we want to have a single variation for all
                   bondtypes, we express it with the word 'global' after the
                   underscore, i.e. 'theta_global'. The only exception is the
                   hyperparameter 'sigma', which is always global and has to be
                   passed as just 'sigma'.

        validmode: How to calculate the RMSE. If 'cross' (default), use function
                   self.cross_validation(). If 'direct', use function
                   self.direct_validation(validfile).

        k_cross: if None (default), we set it to 2

        validfile: if validmode='direct', this variable must not be None
                   (default)

        update: if True (default), training is performed with the best 
                hyperparameters (smallest RMSE) at the end

        output
        ------
        dictionary with results
        '''

        if validmode=='cross':
            if k_cross is None:
                k_cross = 2
            self.gen_crossindcs(k_cross)
        elif validmode=='direct':
            if validfile is None:
                raise ValueError('please specify validfile')

        # build grid
        hyperkeys = hyperdict.keys()
        list_of_sweepvalues = [hyperdict[key] for key in hyperkeys]
        hypergrid = [x for x in itertools.product(*list_of_sweepvalues)]

        # is X_pseudo in hyperkeys? then we need to update self.M in the 
        # process
        update_M = False
        for hypk in hyperkeys:
            if hypk.startswith('X_pseudo'):
                update_M = True

        # perform a prediction on the validation data for every hypergridpoint
        RMSE = []
        for grdpnt in hypergrid:
            # set the hyperparameters
            for i, hypk in enumerate(hyperkeys):
                if hypk == 'sigma':
                    self.sigma = grdpnt[i]
                else:
                    cols = hypk.split('_')
                    bond = cols[-1]
                    param = '_'.join(cols[:-1])
                    if bond == 'global':
                        for b in self.bond_dict.keys():
                            self.bond_dict[b][param] = grdpnt[i]
                    else:
                        self.bond_dict[bond][param] = grdpnt[i]
            
            # update self.M
            if update_M:
                self.M = 0
                for bond_type in self.bond_dict.keys():
                    M =  len(self.bond_dict[bond_type]['X_pseudo'])
                    self.bond_dict[bond_type]['M'] = M
                    self.M += M

            # validation
            if validmode=='cross':
                RMSE += [self.cross_validation()]
            elif validmode=='direct':
                self.train()
                RMSE += [self.direct_validation(validfile)]
        
        # extract best hyperparameters
        RMSE = np.array(RMSE)
        min_indx = np.argmin(RMSE)
        min_RMSE = RMSE[min_indx]
        min_grdpnt = hypergrid[min_indx]

        # update the model
        if update:
            if update_M:
                # update self.M
                self.M = 0
                for bond_type in self.bond_dict.keys():
                    M =  len(self.bond_dict[bond_type]['X_pseudo'])
                    self.bond_dict[bond_type]['M'] = M
                    self.M += M
            for i, hypk in enumerate(hyperkeys):
                if hypk == 'sigma':
                    self.sigma = min_grdpnt[i]
                else:
                    cols = hypk.split('_')
                    bond = cols[-1]
                    param = '_'.join(cols[:-1])
                    if bond == 'global':
                        for b in self.bond_dict.keys():
                            self.bond_dict[b][param] = min_grdpnt[i]
                    else:
                        self.bond_dict[bond][param] = min_grdpnt[i]
            self.train()
                    

        result_dict = {'min_RMSE':min_RMSE,
                       'min_grdpnt':min_grdpnt,
                       'hypergrid':hypergrid,
                       'RMSE':RMSE,
                       'hyperkeys':hyperkeys}

        return result_dict

    def learning_curve(self, num_targets, reset=True):
        '''
        Use a random subset of the training targets for training and predict
        on the rest; calculate the RMSE. Do this for a series of subset sizes
        to calculate a learning curve -> how fast does the model saturate?

        parameters
        ----------
        num_targets: list of subset sizes, e.g [10, 100, 1000]
        reset: if True, then reset the original training data and train afterwards
        '''
        
        RMSE = []
        alpha = []
        R = []
        
        # copy the original training data
        Y_org = copy(self.Y)
        B_org = copy(self.B)
        X_org = copy(self.X)
        D_org = copy(self.D)

        # iterate over different training set sizes
        for i in num_targets:
            indcs = np.arange(len(X_org)) # total force (not component!) indices
            np.random.shuffle(indcs)
            train_indcs = indcs[:i]
            train_indcs_comp = np.array([], dtype=np.int64)
            for j in train_indcs:
                train_indcs_comp = np.append(train_indcs_comp, 
                                             [3*j, 3*j+1, 3*j+2])

            # set the training data variables and train
            self.B = B_org[train_indcs]
            self.X = X_org[train_indcs]
            R += [self.X]
            self.Y = Y_org[train_indcs_comp]
            self.D = D_org[train_indcs_comp]
            self.N = len(self.Y)

            self.train()

            alpha += [copy(self.alpha)]

            residual = self.system_2bF_predict(B_org, X_org, D_org) - Y_org
            RMSE += [np.sqrt(np.sum(residual**2)/len(residual))]

        # reset the training data and train
        if reset:
            self.Y = Y_org
            self.B = B_org
            self.X = X_org
            self.D = D_org
            self.N = len(self.Y)
            self.train()

        return (RMSE, R, alpha)


        



###############################################################################
# IV. Covariance functions and derivatives                                    #
###############################################################################
# in covariance.so (compiled cython code), source code in covariance.pyx

###############################################################################
# IV. Input/Output                                                            #
###############################################################################
    def read_training_data_1bE(self, filename):
        """
        read in training data as prepared by readev (hirshfeld)
        """
        with open(filename, 'r') as train_dat:

            # N total/xc energies
            Y = [] # [E_1, E_2, E_3, ..., F_N]

            # elements
            B = [] # [[B_1_1, B_1_2, ..., B_1_M1],
                   #        ...
                   #  [B_N_1,        ..., B_N_MN]]

            # charges
            X = [] # [[Q_1_1, Q_1_2, ..., Q_1_M1],
                   #        ...
                   #  [Q_N_1,        ..., Q_N_MN]]

            num_atoms = train_dat.readline()
            while num_atoms != '':
                num_atoms = int(num_atoms)
                E_line = train_dat.readline().split()
                # Etot = 2, Exc = 3, Ex = 4, Ec = 5 
                E  = float(E_line[1])
                Y += [E] 
                B123 = []
                X123 = []
                for j in range(num_atoms):
                    D_line = train_dat.readline().split()
                    # read atom type
                    at = D_line[0]
                    if not (at in self.bond_dict.keys()):
                        continue
                    # read charge
                    q = float(D_line[1])
                    X123 += [q]
                    B123 += [at]
                B += [np.array(B123)]
                X += [np.array(X123)]
                num_atoms = train_dat.readline()

            return np.array(Y), np.array(B), np.array(X)


    def read_training_data_2bF(self, filename):
        """
        read in training data as prepared by get_distances.py
        """
        with open(filename, 'r') as train_dat:

            # N total force components
            Y = [] # [F_1_1, F_1_2, F_1_3, F_2_1, F_2_2, F_2_3, ..., F_3_N/3]

            # bond types
            B = [] # [[B_1_1, B_1_2, ..., B_1_M1],
                   #        ...
                   #  [B_N/3_1, ..., B_N/3_MN/3]]

            # bond lengths
            X = [] # [[R_1_1, R_1_2, ..., R_1_M1],
                   #        ...
                   #  [R_N/3_1, ..., R_N/3_MN/3]]

            # bond vectors
            D = [] # [[x_1_1_1, x_1_1_2, ..., x1_1_M1],
                   #             ...
                   #  [x_3_N/3_1, ..., x_3_N/3_MN/3]]

            num_bonds = train_dat.readline()
            while num_bonds != '':
                num_bonds = int(num_bonds)
                F_line = train_dat.readline().split()
                fx = float(F_line[1])
                fy = float(F_line[2])
                fz = float(F_line[3])
                Y += [fx,fy,fz]
                D1 = []
                D2 = []
                D3 = []
                B123 = []
                X123 = []
                for j in range(num_bonds):
                    # only consider bonds, which lie inside cutoff
                    # note: changing the cutoff attribute of an gp object
                    # to a larger value than the initial one, will not reload
                    # the bond lists automatically! You will have to do
                    # this by hand.
                    D_line = train_dat.readline().split()
                    # bonds AB and BA are equivalent, use symbol given in 
                    # bond_dict for this bond
                    b = D_line[0]
                    b2 = b.split('-')[1] + '-' + b.split('-')[0]
                    if not (b in self.bond_dict.keys() or b2 in self.bond_dict.keys()):
                        continue
                    if not b in self.bond_dict.keys():
                        b = b2
                    # throw out bonds longer than the cutoff
                    R = float(D_line[1])
                    if R > self.bond_dict[b]['cutoff']:
                        continue
                    X123 += [R]
                    B123 += [b]
                    # determine the gradients of the distance
                    dx = float(D_line[2])
                    dy = float(D_line[3])
                    dz = float(D_line[4])
                    D1 += [dx/R]
                    D2 += [dy/R]
                    D3 += [dz/R]
                B += [np.array(B123)]
                X += [np.array(X123)]
                D += [np.array(D1), np.array(D2), np.array(D3)]
                num_bonds = train_dat.readline()

            return np.array(Y), np.array(B), np.array(X), np.array(D)


###############################################################################
# V. Convert to spline and write to .skf file                                 #
###############################################################################

    def exp_func(self, x, a1, a2, a3):
        return np.exp(-a1*x+a2) + a3

    def make_skf_spline(self, s0, s1,  N, outputfile, bond_type=None,
                        cutoff=None):
        """
        We fit the GP predictive function with an exponential function
        from 0. to s1 with data between s0 and s1  and with cubic splines
        between s1 and cutoff.
        Formally a degree five polynomial is needed instead of a cubic spline
        for the last section before the cutoff, so when writing to the .skf
        file we set the corresponding coefficients c4 and c5 to zero.

        for an explanation of the format, see
        http://www.dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf

        units of .skf file are Bohr and Hartree!

        """

        if bond_type is None:
            bond_type = self.bond_types[0]

        if cutoff is None:
            cutoff = self.bond_dict[bond_type]['cutoff']

        exp_range = np.linspace(s0, s1, 100.)
        exp_data = -self.pot_vpredict(exp_range, bond_type)
        exp_fit_result = curve_fit(self.exp_func, exp_range/Bohr, exp_data/Hartree,maxfev = 1000)
        a1, a2, a3 = exp_fit_result[0]

        cube_edges = np.linspace(s1, cutoff, N+1)
        dx = (cube_edges[1] - cube_edges[0])/Bohr

        # M*a = b, with a the spline coefficients

        # for an explanation of boundary conditions of the splines see last slide of 
        # http://users.ph.tum.de/srecksie/lehre/CPI-1516/notes/2015-11-24-Note-15-33.pdf

        # we have N+1 data points and N cubic spline functions, so we need
        # to determine 4*N parameters
        M = np.zeros((4*N, 4*N))
        b = np.zeros(4*N)

        # offsets
        for i in range(N):
            M[i,4*i] = 1.
        b[0:N] = -self.pot_vpredict(cube_edges[:-1], bond_type)/Hartree
        b[0] = self.exp_func(s1/Bohr, a1, a2, a3) # connection to exp_func

        # continuity of curve
        const_curve = np.array([1., dx, dx**2, dx**3])
        for i in range(N):
            M[N+i,4*i:4*i+4] = const_curve
        b[N:2*N] = -self.pot_vpredict(cube_edges[1:], bond_type)/Hartree

        # continuity of first derivative
        const_der1 = np.array([1., 2*dx, 3*dx**2, 0., -1.])
        for i in range(0, N-1):
            M[2*N+i, 4*i+1:4*i+6] = const_der1

        # continuity of second derivative
        const_der2 = np.array([1., 3.*dx, 0., 0., -1.])
        for i in range(0, N-1):
            M[3*N-1+i, 4*i+2:4*i+7] = const_der2

        # fixing second derivatives at endpoints of cubic splines
        M[4*N-2, 2] = 1.
        b[4*N-2] = a1**2/2. * np.exp(-a1*s1/Bohr + a2)
        M[4*N-1, -2] = 1.

        # solve the matrix equation
        a = np.linalg.solve(M, b)

        # write to outputfile in .skf compatible format
        with open(outputfile, 'w') as out:
            out.write('Spline\n')
            out.write('{} {}\n'.format(N, cutoff/Bohr))
            out.write('{} {} {}\n'.format(a1, a2, a3))
            for i in range(N-1):
                out.write('{} {} {} {} {} {}\n'.format(cube_edges[i]/Bohr,
                                                       cube_edges[i+1]/Bohr,
                                                       a[4*i],
                                                       a[4*i+1],
                                                       a[4*i+2],
                                                       a[4*i+3]))

            out.write('{} {} {} {} {} {} {} {}\n'.format(cube_edges[N-1]/Bohr,
                                                       cube_edges[N]/Bohr,
                                                       a[-4],
                                                       a[-3],
                                                       a[-2],
                                                       a[-1],
                                                       0.,
                                                       0.))
#EOF
