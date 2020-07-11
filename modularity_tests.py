from tvb.simulator.lab import *
from tvb.datatypes import time_series
from tvb.basic.config import settings
import matplotlib.pyplot as plt
import os as os
import numpy as np
import numpy.linalg as nplinalg
import scipy.signal as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def positive(x):
    for i in x:
        if i < 0:
            return False
    return True

def power_iteration(X, thresh):

    step_i = np.random.rand(X.shape[0])
    apply = np.dot(X, step_i)
    step_iplus1 = apply/(nplinalg.norm(apply))
    i = 0
    
    while nplinalg.norm(step_iplus1 - step_i) > thresh:
        
        step_i = step_iplus1
        apply = np.dot(X, step_i)
        step_iplus1 = apply/(nplinalg.norm(apply))
        print(i)
        i = i+1

    E_value = nplinalg.norm(np.dot(X, step_i))
        
    return step_i, E_value

def adjacency_matrix(X):

    adj = X > 0
    adj = adj*1

    return adj

#for first split
def modulate(X):
    
    numnodes = X.shape[0]
    #define quantities m, k(in)_i, k(out)_j, B + B_T
    num_edges = np.sum(X)
    in_degrees = np.sum(X, axis = 1)
    out_degrees = np.sum(X, axis = 0)
    B = X - np.outer(in_degrees, out_degrees)/num_edges
    B_T = B.transpose()
    B_1 = B + B_T
    print(B_1)

    network_modularity_properties = {'numnodes':numnodes, 'num_edges':num_edges,
                                     'B_1':B_1}

    #find eigenvector with largest eigenvalue of B + B_T
    e_vals, e_vects = nplinalg.eig(B_1)
    eval_ind = np.where(e_vals == np.max(e_vals))[0]
    e_val = e_vals[eval_ind]
    e_vect = e_vects[:, eval_ind]
    e_vect = e_vect.transpose()
    
    #find s that(approximately) maximises Q
    s_pos = adjacency_matrix(e_vect)
    s_neg = -(np.ones(numnodes) - s_pos)
    s = s_pos + s_neg

    #output s, Q, community indices, and network properties (for subdivision)
    s = s.transpose()
    Q = 1/(4*num_edges)*np.dot(s.transpose(), np.dot(B_1, s))
    group_1 = np.nonzero(s_pos)[1]
    group_2 = np.nonzero(s_neg)[1]
    
    return Q, [group_1, group_2], network_modularity_properties

#for subsequent splits
def modulate_subdivide(group, NMP):

    #recall NMPs, define B_g
    numnodes = NMP['numnodes']
    num_edges = NMP['num_edges']
    B_1 = NMP['B_1']
    B_g = B_1[np.ix_(group, group)] - np.diag(np.sum(B_1[np.ix_(group, group)], axis = 1))

    #find eigenvector with largest eigenvalue of B_g
    e_vals, e_vects = nplinalg.eig(B_g)
    eval_ind = np.where(e_vals == np.max(e_vals))[0]
    e_val = e_vals[eval_ind]
    e_vect = e_vects[:, eval_ind]
    e_vect = e_vect.transpose()
    
    #find s that (approximately) maximises delta_Q
    s_pos = adjacency_matrix(e_vect)
    s_neg = -(np.ones(len(group)) - s_pos)
    s = s_pos + s_neg

    #output s, delta_Q, community indices
    s = s.transpose()
    delta_Q = 1/(4*num_edges)*np.dot(s.transpose(), np.dot(B_g, s))
    group_1 = group[np.nonzero(s_pos)[1]]
    group_2 = group[np.nonzero(s_neg)[1]]
    
    return delta_Q, [group_1, group_2]

fivetwelve_node = 'geombinsurr_partial_000_Gs.zip'
seventysix_node = 'connectivity_76.zip'
ninetysix_node = 'connectivity_96.zip'
oneninetytwo_node = 'connectivity_192.zip'

white_matter = connectivity.Connectivity.from_file(fivetwelve_node)
weights = white_matter.weights
numnodes = weights.shape[0]
positions = white_matter.centres.transpose()
x = positions[0]
y = positions[1]
z = positions[2]
          
Q, groups, NMP = modulate(weights)
delta_Q_1, groups[0] = modulate_subdivide(groups[0], NMP)
delta_Q_2, groups[1] = modulate_subdivide(groups[1], NMP)
groups = groups[0] + groups[1]
delta_Q = delta_Q_1[0][0] + delta_Q_2[0][0]
Q = Q[0][0] + delta_Q
print(Q)

print(delta_Q)
if delta_Q < 0:
    
    Q, groups, NMP = modulate(weights)

else:
    while delta_Q > 0.03:

        delta_Qs = []
        new_groups = []
        
        for i in range(len(groups)):
            
            delta_Q_i, groups_i = modulate_subdivide(groups[i], NMP)
            delta_Qs.append(delta_Q_i)
            new_groups.append(groups_i[0])
            new_groups.append(groups_i[1])
        
        delta_Qs = np.array(delta_Qs)
        delta_Q = np.sum(delta_Qs)
        
        if delta_Q > 0:
            
            groups = new_groups
            Q = Q + delta_Q
            
        print(delta_Q)
    
    
print(Q)
print(groups)
s = np.zeros(numnodes)
new_groups = []

for i in range(len(groups)):
    if not len(groups[i]) == 0:
        new_groups.append(groups[i])

groups = new_groups

for i in range(len(groups)):
    
    s[groups[i]] = i*np.ones(len(groups[i]))
    
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(x, y, z, c = s, cmap = 'coolwarm', s = 100)
plt.show()

    
    
    

