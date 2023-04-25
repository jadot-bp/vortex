#!/usr/bin/env python
# coding: utf-8

# In[2]:


import lyncs_io as io
import numpy as np
import time
import itertools
import os

import pandas as pd

import gluon_utils as gu

def q_wilson(q_hat,a=1):
    return (2/a)*np.sin(np.asarray(q_hat)*a/2)
    #return np.asarray(q_hat)
            
def q_improved(q_hat,a=1):
    return (2/a)*np.sqrt( np.sin(np.asarray(q_hat)*a/2)**2 + (1/3)*np.sin(np.asarray(q_hat)*a/2)**4 )

def get_qhat(coord,shape,a=1):
        """Calculates q^_\mu"""
        
        return (2*np.pi/a)*np.asarray(coord)/np.asarray(shape)

def unique_permute(coord):
    
    items = list(itertools.permutations(coord))
   
    unique_items = []
    
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
            
    return np.asarray(unique_items)
    
def spatial(Nt,Nc,t0,check_divA=False,calculate_z3=False,return_z3=False,rand_selection=True,save_prop=True, regenerate=True):
    """Calculates the spatial gluon propagator at time t0.

    Parameters:
        Nt: Temporal extent
        Nc: Number of gauge configurations to sample
        t0: Time slice
    Optional Parameters:
        check_divA [False]: Calculate the value of |div(A)|^2 for each configuration
        calculate_z3 [False]: Calculate the Z3-averaged propagator
        return_z3 [False]: Return the Z3-averaged propagator instead of the standard
        rand_selection [True]: Iterate through configurations randomly when sampling
        save_prop [True]: Save calculated values of the propagator
        
    Returns:
        q: Array of coordinates
        D_results: Array of propagator values
    """
    D_results = []

    available_transforms = []
    
    for file in os.listdir(f"../gauge_confs/transforms/{Nt}x32"):
        if file.endswith(".gauge.lime"):
            base_name = file.rstrip(".gauge.lime")
            available_transforms.append(base_name)
            
    if rand_selection:
        selection = np.random.choice(available_transforms,size=Nc,replace=False)
    else:
        selection = np.arange(1,Nc+1)
    
    for n in range(Nc):
        #input_file = "Gen2l_64x32n1.lime"
        input_file = f"../gauge_confs/samples/{Nt}x32/{selection[n]}"
        gauge_file = f"../gauge_confs/transforms/{Nt}x32/{selection[n]}.gauge.lime"

        gauge_output = f"../gauge_confs/props/Nt{Nt}/{selection[n]}_t0_{t0}.prop"
        z3_output = f"../gauge_confs/props/Nt{Nt}/{selection[n]}_t0_{t0}.prop.z3"
        
        file_found = False
        
        if os.path.exists(gauge_output) and not regenerate:
            print("Cached file found. Loading...")
            prop = pd.read_csv(gauge_output).values[:]
            q = prop[:,:4]
            results = prop[:,4]
            D_results.append(results)
            file_found = True
        if os.path.exists(z3_output) and not regenerate:
            print("Cached Z3 file found. Loading...")
            prop = pd.read_csv(gauge_output).values[:]
            q = prop[:,:4]
            results = prop[:,4]
            D_results.append(results)
            file_found = True
        
        if not file_found:
            print("No cached file found. Generating...")
        
            data = io.load(input_file, format="openqcd")
            gauge = io.load(gauge_file, format="lime")

            gf = gu.lattice(data,(Nt,32,4,3))
                
            # Apply gauge transformation
            gf.apply_gauge(gauge)

            if check_divA:
                # check divA
                print("div.A:",gf.evaluate_divA())
        
            # Transform to Fourier space
            gf.transform(axes=(0,1,2,3))

            results = []
            z3_results = []
            
            q = []
            z3_q = []
        
            # Loop over t,qx,qy,qz


            for t in [t0]:
                for qx in range(gf.shape[1]//4):
                    for qy in range(gf.shape[2]//4):
                        for qz in range(gf.shape[3]//4):
                            D = []

                            # If Z3 averaging, calculate equivalent coordinates
                            if qz >= qy and qy >= qx and calculate_z3:
                                coords = unique_permute([qx,qy,qz])
                            else:
                                coords = [[qx,qy,qz]]

                            for coord in coords:
                                # Calculate propagator
                                
                                D_sum = 0
                                for mu in [0,1,2,3]:
                                    # Decompose A in terms of Gell-Mann components
                                    A = gu.decompose_su3(gf.get_A((t,coord[0],coord[1],coord[2]),mu))
                                    A_neg = gu.decompose_su3(gf.get_A((-t,-coord[0],-coord[1],-coord[2]),mu))

                                    D_sum += np.dot(A,A_neg)
                                D.append(D_sum)

                            # Save results if Z3
                            if qz >= qy and qy >= qx and calculate_z3:
                                # Perform Z3 averaging
                                z3_q.append([t,qx,qy,qz])
                                z3_results.append(np.sum(D)/len(coords))
                                
                                # Retain un-averaged D
                                q.append([t,qx,qy,qz])
                                results.append(np.asarray(D)[np.all(coords == [qx,qy,qz],axis=1)][0])
                            
                            else:
                                q.append([t,qx,qy,qz])
                                results.append(np.sum(D))

            q = np.asarray(q)
            z3_q = np.asarray(z3_q)
            
            results = np.asarray(results)
            z3_results = np.asarray(z3_results)

            # Multiply propagator by prefactors
            results_cleaned = []
            
            for pos,result in enumerate(results):
                if np.all(np.asarray(q)[pos] == 0):
                    results_cleaned.append(result*2/((gf.Nc**2-1)*gf.Nd*gf.V))
                else:
                    #results_cleaned.append([np.linalg.norm(q_improved(result[0])),result[1]*2/((gf.Nc**2-1)*(gf.Nd-1)*gf.V)])
                    results_cleaned.append(result*2/((gf.Nc**2-1)*(gf.Nd-1)*gf.V))

            results_cleaned = np.asarray(results_cleaned)
            
            if calculate_z3:
                z3_results_cleaned = []
            
                for pos,result in enumerate(z3_results):
                    if np.all(np.asarray(z3_q)[pos] == 0):
                        z3_results_cleaned.append(result*2/((gf.Nc**2-1)*gf.Nd*gf.V))
                    else:
                        z3_results_cleaned.append(result*2/((gf.Nc**2-1)*(gf.Nd-1)*gf.V))

                z3_results_cleaned = np.asarray(z3_results_cleaned)
            
            if return_z3:
                D_results.append(z3_results_cleaned.real)       
            else:
                D_results.append(results_cleaned.real)
            print(f"{input_file} done.")
        
            # Save propagator values
            if save_prop:
                print("Saving propagator...")
                out_df = pd.DataFrame(np.hstack([q.real,results_cleaned.reshape(-1,1).real]))
            
                out_df.to_csv(gauge_output,index=None,header=['qt','qx','qy','qz','D(q)_s'])
            if save_prop and calculate_z3:
                print("Saving Z3-averaged propagator...")
                out_df = pd.DataFrame(np.hstack([z3_q.real,z3_results_cleaned.reshape(-1,1).real]))
            
                out_df.to_csv(z3_output,index=None,header=['qt','qx','qy','qz','z3_D(q)_s'])
    
    if return_z3:
        return z3_q.real, D_results
    else:
        return q.real, D_results
        
if __name__ == "__main__":
    import sys
    spatial(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
