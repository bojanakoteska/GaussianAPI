#!/home/artur/anaconda2/bin/python

# Script takes xyz file as input and extracts the pairwise absolute distances 
# and distance vectors.

import sys
import numpy as np
import itertools
import getopt

def main(argv):
    cutoff = None
    output_filename = 'training.dat'

    opts, args = getopt.getopt(argv, 'c:o:', ['cutoff=', 'output_filename='])

    for opt, arg in opts:
        if opt in ('-c', '--cutoff'):
            cutoff = float(arg)
        elif opt in ('-o', '--output_filename'):
            output_filename = arg

    xyz_filename = ''.join(args) # filename of xyz file
    
    if cutoff is None:
        print 'You have not specified a cutoff. cutoff is set to 10.'
        cutoff = 10.

    # determine number of structures in xyz file
    num_structures = 0
    with open(xyz_filename, 'r') as xyz:
        line = xyz.readline()
        while line != '':
            num_structures += 1
            num_atoms = int(line)
            for i in range(num_atoms+2):
               line = xyz.readline()
    
    # extract pairwise distances and total forces in an ordered manner
    data = []
    # we create a dictionary: {bond_symbol:bond_distances}, to count all 
    # bond distances in the training data (only once)
    bond_dists_dict = {}
    with open(xyz_filename, 'r') as xyz:
        for structure in range(num_structures):
            symbols = []
            atom_positions = []
            forces = []
            num_atoms = int(xyz.readline())
            comm_line = xyz.readline() # comment line
            # get lattice vectors from comm_line
            cl = comm_line.split()
            l1x = float(cl[0].split('="')[1])
            l1y = float(cl[1])
            l1z = float(cl[2])
            l2x = float(cl[3])
            l2y = float(cl[4])
            l2z = float(cl[5])
            l3x = float(cl[6])
            l3y = float(cl[7])
            l3z = float(cl[8].split('"')[0])
            l1 = np.array([l1x, l1y, l1z])
            l2 = np.array([l2x, l2y, l2z])
            l3 = np.array([l3x, l3y, l3z])
            # get periodic boundary conditions from comm_line
            for i, string in enumerate(cl):
                if string.startswith('pbc'):
                    pbc_index = i
                    break
            pbc = [cl[pbc_index][-1], cl[pbc_index+1], cl[pbc_index+2][0]]
            #pbc = ['F', 'F', 'F']
            # grid for bond search search via translation
            # IS IT ACTUALLY WELL DEFINED???
            n1, n2, n3 = 0, 0, 0
            if pbc[0] == 'T':
                n1 = max(2*int(cutoff/np.linalg.norm(l1)), 2)
            if pbc[1] == 'T':
                n2 = max(2*int(cutoff/np.linalg.norm(l2)), 2)
            if pbc[2] == 'T':
                n3 = max(2*int(cutoff/np.linalg.norm(l3)), 2)
            T = [] # list of translation vectors (including [0,0,0])
            for a1 in range(-n1,n1+1):
                for a2 in range(-n2, n2+1):
                    for a3 in range(-n3, n3+1):
                        trans_vector = a1*l1 + a2*l2 + a3*l3
                        T += [trans_vector]
            clone_bonds = []
            clone_bond_dists = []
            for trans_vector in T:
                if not trans_vector.any(): # do not consider [0,0,0]
                    continue
                R = np.linalg.norm(trans_vector)
                if R < cutoff:
                    clone_bonds += [trans_vector]
                    clone_bond_dists += [R]
            # print bond_dists
            # get atom symbols, positions and forces
            for al in range(num_atoms):
                al = xyz.readline()
                al = al.split()
                # atom symbol must be first!
                symbols += [al[0]]
                # atom positions must start after atom symbol!
                x = float(al[1])
                y = float(al[2])
                z = float(al[3])
                atom_positions += [[x, y, z]]
                # total forces must start after atom positions!
                fx = float(al[4])
                fy = float(al[5])
                fz = float(al[6])
                forces += [[fx, fy, fz]]
            # turn atom_positions and forces into numpy 2D arrays
            atom_positions = np.array(atom_positions)
            forces = np.array(forces)
            
            # get list of bond symbols
            unique_symbols = []
            for s in symbols:
                if not s in unique_symbols:
                    unique_symbols += [s]
            bond_symbols = []
            for s in unique_symbols:
                bond_symbols += [s+'-'+s]
            num_symbols = len(unique_symbols)
            for i in range(num_symbols):
                for j in range(i, num_symbols):
                    if j+1 < num_symbols:
                        bond_symbols += [
                                     unique_symbols[i]+'-'+unique_symbols[j+1]]
            for bs in bond_symbols:
                bs2 = bs.split('-')[1]+'-'+bs.split('-')[0]
                if not (bs in bond_dists_dict.keys() or bs2 in bond_dists_dict.keys()):
                    bond_dists_dict[bs] = []
                if bs == bs2:
                    bond_dists_dict[bs] += clone_bond_dists # bonds: atom-clone

            # for every total force on an atom i, calculate pairwise distances
            # to every bond partner of i (including i itself) and accept as a 
            # valid bond, if it lies inside the cutoff sphere
            bonds = []
            for i in range(num_atoms):
                symbol_i = symbols[i] # symbol of atom i
                bonds_i = [] # all bonds of atom i inside cutoff.
                             # the j-th element of bonds_i is bonds_ij.
                             # bonds_ij is a list of bonds, because j refers
                             # to atom j + its translations by the lattice 
                             # vectors. every element of bonds_ij is 
                             # a tuple: (distance_vector, bond_type)
               
                pos_i = atom_positions[i,:] # position vector of atom i
                
                # bonds from atom i to atoms j with j<i
                # using symmetry: bond(i-->j) = -bond(j-->i)
                for j in range(i):
                    bonds_ij = []
                    for bond in bonds[j][i]: # might be empty
                        bonds_ij += [(-bond[0], symbol_i+'-'+symbols[j])]
                    bonds_i += [bonds_ij]
                
                # bonds from i to i, 
                # i.e. its 'lattice vector translated clones'.
                # (assign bond_types to clone bonds before adding)
                clone_bonds_symbolized = []
                for clone_bond in clone_bonds:
                    clone_bonds_symbolized += [(clone_bond, 
                                                symbol_i+'-'+symbol_i)]
                bonds_i += [clone_bonds_symbolized]
                
                # bonds from i to j with j>i
                for j in range(i+1, num_atoms):
                    bonds_ij = []
                    dist_vector = pos_i - atom_positions[j,:]
                    # additional neighbors inside cutoff via translation by 
                    # lattice vectors: 'original' dist_vector will appear in the
                    # following grid search (because [0,0,0] is in T) and is 
                    # also dismissed if >= cutoff.
                    for t in T:
                        trans_dist_vector = dist_vector + t
                        R = np.linalg.norm(trans_dist_vector)
                        if R < cutoff:
                            bs = symbol_i+'-'+symbols[j]
                            bonds_ij += [(trans_dist_vector, bs)]
                            if bs in bond_dists_dict.keys():
                                bond_dists_dict[bs] += [R]
                            else:
                                bond_dists_dict[symbols[j]+'-'+symbol_i] += [R]
                    bonds_i += [bonds_ij]
                bonds += [bonds_i]
            data += [(num_atoms, forces, bonds)]
    # write data to textfile
    with open(output_filename, 'w') as out:
        for tup in data:
            num_atoms = tup[0]
            for i in range(num_atoms):
                force_i = tup[1][i,:]
                # we dissolve the nested structure of bonds_i: 
                # every element of bonds_i shall be a direct list of all bonds
                # in which i participates
                bonds_i = list(itertools.chain(*tup[2][i]))
                num_bonds_i = len(bonds_i)
                out.write('{}\n'.format(num_bonds_i))
                out.write('F   {: 20.16f} {: 20.16f} {: 20.16f}\n'.format(force_i[0],
                                                                   force_i[1],
                                                                   force_i[2]))
                for bond in bonds_i:
                    R = np.linalg.norm(bond[0])
                    out.write('{}  {: .12f} {: .12f} {: .12f} {: .12f}\n'.format(
                                                               bond[1],
                                                               R,
                                                               bond[0][0],
                                                               bond[0][1],
                                                               bond[0][2]))
    # write bond distances to textfile
    with open('bond_dists_{}'.format(output_filename), 'w') as out:
        for bs in bond_dists_dict.keys():
            sorted_bond_dists = np.sort(np.array(bond_dists_dict[bs]))
            out.write('{}\n'.format(bs))
            for bd in sorted_bond_dists:
                out.write('{}\n'.format(bd))
            out.write('\n')


if __name__ == "__main__":
    main(sys.argv[1:])
