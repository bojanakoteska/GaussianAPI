#!/data/guest5/programs/anaconda2/bin/python

# define function based on spline coefficients from a .skf file
# see http://www.dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf

import numpy as np
import spline_predict
from ase.units import Bohr, Hartree

class Spline:
    
    def __init__(self, bond_dict={}):
        """
        read in spline coefficients
        """
        # bond_dict = {'A-A':{'skf':'path/to/A-A.skf'}, 
        #              'B-B':{'skf':'path/to/B-B.skf'},
        #              'A-B':{'skf':'path/to/A-B.skf'}}
        self.bond_dict = bond_dict
        self.bond_types = []
        for bond_type in bond_dict.keys():
            self.bond_types += [bond_type]

        for bond_type in self.bond_types:
            # extract spline coefficients from .skf file
            skf_file = self.bond_dict[bond_type]['skf']
            with open(skf_file, 'r') as skf:
                line = skf.readline()
                while not line.startswith('Spline'):
                    line = skf.readline()
                first_line = skf.readline().split()
                nInt = int(first_line[0])
                cutoff = float(first_line[1])
                second_line = skf.readline().split()
                a1 = float(second_line[0])
                a2 = float(second_line[1])
                a3 = float(second_line[2])
                cubic_coeff = []
                cubic_intervls = []
                for i in range(nInt-1):
                    cubic_line = skf.readline().split()
                    cubic_intervls += [(float(cubic_line[0]),
                                        float(cubic_line[1]))]
                    cubic_coeff += [[float(cubic_line[2]),
                                     float(cubic_line[3]),
                                     float(cubic_line[4]),
                                     float(cubic_line[5])]]
                cubic_intervls = np.array(cubic_intervls)
                cubic_coeff = np.array(cubic_coeff)
                s1 = cubic_intervls[0][0] # border exp to cubic
                s2 = cubic_intervls[-1][1] # border cubic to last
                last_coeff = []
                last_line = skf.readline().split()
                last_coeff += [float(last_line[2]),
                               float(last_line[3]),
                               float(last_line[4]),
                               float(last_line[5]),
                               float(last_line[6]),
                               float(last_line[7])]
                last_coeff = np.array(last_coeff)
            
            # write spline coefficients to bond_dict
            self.bond_dict[bond_type]['cutoff'] = cutoff
            self.bond_dict[bond_type]['a1'] = a1
            self.bond_dict[bond_type]['a2'] = a2
            self.bond_dict[bond_type]['a3'] = a3
            self.bond_dict[bond_type]['cubic_coeff'] = cubic_coeff
            self.bond_dict[bond_type]['cubic_intervls'] = cubic_intervls
            self.bond_dict[bond_type]['s1'] = s1
            self.bond_dict[bond_type]['s2'] = s2
            self.bond_dict[bond_type]['last_coeff'] = last_coeff

        # initialization of potential and force functions from spline_predict.so
        # these have Hartree and Bohr as units!
        self.potential = spline_predict.Potential()
        self.force = spline_predict.Force()
        
        # vectorization of prediction function
        self.vpredict = np.vectorize(self.predict)

    def predict(self, r, bond_type=None, der=0):
        """
        r must be in Angstrom
        returns potential in eV and force in eV/Angstrom
        """

        if bond_type is None:
            bond_type = self.bond_types[0]

        # unpacking spline coefficients
        cutoff = self.bond_dict[bond_type]['cutoff'] 
        a1 = self.bond_dict[bond_type]['a1']
        a2 = self.bond_dict[bond_type]['a2']
        a3 = self.bond_dict[bond_type]['a3']
        cubic_coeff = self.bond_dict[bond_type]['cubic_coeff']
        cubic_intervls = self.bond_dict[bond_type]['cubic_intervls']
        s1 = self.bond_dict[bond_type]['s1']
        s2 = self.bond_dict[bond_type]['s2']
        last_coeff = self.bond_dict[bond_type]['last_coeff']
        
        if der == 0:
            return self.potential.evaluate(r/Bohr, s1, s2, cutoff,
                                           a1, a2, a3,
                                           cubic_coeff,
                                           cubic_intervls,
                                           last_coeff) * Hartree
        if der == 1:
            return self.force.evaluate(r/Bohr, s1, s2, cutoff,
                                       a1, a2, a3,
                                       cubic_coeff,
                                       cubic_intervls,
                                       last_coeff) * Hartree/Bohr
    
    def system_2bF_predict(self, B_val, X_val, D_val):
        """
        does the same job as system_2bF_predict of the gp_sparse_linear module,
        but with spline data instead of a trained Gaussian process.
        returns forces in eV/Angstrom
        """
    
        return (spline_predict.system_2bF_predict(self.force,
                                                  self.bond_dict,
                                                  B_val, X_val/Bohr, D_val)
                                                  * Hartree/Bohr)
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

