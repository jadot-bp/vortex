#!/usr/bin/env python
# coding: utf-8

import lyncs_io as io
import numpy as np
import time
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd

import gluon_utils as gu
import gluon


def unique_permute(coord):
    """Return all cyclic permutations of coord."""
    
    
    items = list(itertools.permutations(coord))
   
    unique_items = []
    
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
            
    return np.asarray(unique_items)

def get_data(Nt,Z3_avg=False,cylinder=False,cone=False,t_avg=True,c_radius=None,c_angle=None, return_info=False):
    """
    Fetch and process saved gluon propagator data.
    
    Args:
        Nt: Temporal extent of the lattice
        Z_3 [False] : Perform Z_3 averaging of the spatial indices
        cylinder [False]: Perform cylinder cut
        cone [False]: Perform cone cut
        t_avg [False]: Perform averaging over time slices
        c_radius [None] : Radius of the cylinder cut
        return_info [False]: Return info such as bcd signature
    """
    
    Ns = 32
    
    prop_loc = f"/home/ben/Work/gauge_confs/props/Nt{Nt}"
    
    # Get list of file names which satisfy Z_3 and timeslice selections
    prop_names = []

    for prop in os.listdir(prop_loc):
        if prop.endswith('.prop'):
            prop_names.append(prop)
            
    # Get D and q
    D = []

    for prop in prop_names:
        data = pd.read_csv(f"{prop_loc}/{prop}").values[:]
        q_coord = data[:,:4]
        D_prop = data[:,4]
        D.append(D_prop)
    
    # Momentum correction
    q_hat = np.asarray([gluon.get_qhat(i,(Nt,Ns,Ns,Ns)) for i in q_coord])
    q = np.asarray([np.linalg.norm(gluon.q_improved(i[1:])) for i in q_hat])

    D = np.asarray(D)
     
    bcd_sig = np.asarray([np.all(coord[1:] == coord[0]) for coord in q_coord]) # Body-centred diagonal mask
        
    # Perform time slice averaging
    if t_avg:
        print("Averaging time slices...")    
        
        D_averaged = []
        q_averaged = []
        q_coord_averaged = []
        
        for q_ in range(int((Ns//2)**3)):
            q_averaged.append(q[q_])
            q_coord_averaged.append(q_coord[q_,1:])
            D_averaged.append(np.mean(D[:,q_::(Ns//2)**3],axis=1))
        
        q = np.asarray(q_averaged)
        q_coord = np.asarray(q_coord_averaged)
        D = np.asarray(D_averaged).T
        
        bcd_sig = bcd_sig[:int(Ns//2)**3]
        
    # Perform cylinder cut
    if cylinder:
        print("Performing cylinder cut...")
        cylinder_mask = []

        body_norm = np.ones(3)/np.linalg.norm(np.ones(3))

        for coord in q_coord:
            if np.all(coord == coord[0]): # Handle on-diagonal coordinates
                r = 0
            else:
                q_norm = np.linalg.norm(coord)
                theta = np.arccos(np.dot(body_norm,coord)/q_norm)
    
                r = q_norm * np.sin(theta)
        
            cylinder_mask.append(True if r <= c_radius else False)

        cylinder_mask = np.asarray(cylinder_mask, dtype=bool)

        q = q[cylinder_mask]
        q_coord = q_coord[cylinder_mask,:]
        D = D[:,cylinder_mask]
        
        bcd_sig = bcd_sig[cylinder_mask]
    
    # Perform Z3 averaging
        
    if Z3_avg:
        D_averaged = []
        q_averaged = []
        
        bcd_sig = []
        
        counter = 0
        
        while len(q_coord) > 0:
            z3_coords = unique_permute(q_coord[0])
            z3_ids = np.argwhere([np.any(np.all(q_ == z3_coords, axis=1)) for q_ in q_coord]).flatten()

            bcd_sig.append(True) if len(z3_ids) == 1 else bcd_sig.append(False)
            
            q_averaged.append(q_coord[0])
            D_averaged.append(np.mean(D[:,z3_ids],axis=1))
              
            q_coord = np.delete(q_coord, z3_ids, axis=0)
            D = np.delete(D,z3_ids,axis=1)
        
        q_hat = np.asarray([gluon.get_qhat(i,(Ns,Ns,Ns)) for i in np.asarray(q_averaged)])
        q = np.asarray([np.linalg.norm(gluon.q_improved(i)) for i in q_hat])
        
        D = np.asarray(D_averaged).T
        
        bcd_sig = np.asarray(bcd_sig)
    
    if return_info:
        return q, D, bcd_sig
    else:
        return q,D
    
if __name__ == "__main__":
    import sys
    get_data(*sys.argv[1:])