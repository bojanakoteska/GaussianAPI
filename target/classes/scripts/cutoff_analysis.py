import numpy as np
import sys
def cuttof(bond_file,v1,v2):
    with open(bond_file, 'r') as bf:
            bond_dists = []
            for line in bf:
                if line == v1 + '\n' or line == v2 + '\n':
                    break
            for line in bf:
                if line.startswith('\n'):
                    break
                bond_dists += [float(line)]
            bond_dists = np.array(bond_dists)
            #print(bond_dists)
            bond_dists=np.round(bond_dists, 1)
            #print(bond_dists)
            values=[]
            frequences=[]
            i = 0
            s=bond_dists
            while( i < len(s) - 1) :
                count = 1
                while s[i] == s[i + 1] :
                    i += 1
                    count += 1
                    if i + 1 == len(s):
                        break
                #print(s[i],count)
                values.append(s[i])
                frequences.append(count)
                i += 1
            #print(values)
            #print(frequences)
            first_value=values[0]
            #print(first_value)
            j=0
            i=first_value
            first_zero_index=j
            first_zero_value=i
            for x in values:
                if values[j]==i:
                    j=j+1
                    i=round(i+0.1,1)
                else:
                    first_zero_index=j
                    first_zero_value=i
                    break
    return values[first_zero_index]
            
                    
                    
                    
                        
                    
            
            
            
