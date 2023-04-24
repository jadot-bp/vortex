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
    
def spatial(Nt,nc,t0,check_divA=False,z3_avg=True,rand_selection=True,save_prop=True):   
    D_results = []

    available_transforms = []
    
    for file in os.listdir(f"../gauge_confs/transforms/{Nt}x32"):
        if file.endswith(".gauge.lime"):
            base_name = file.rstrip(".gauge.lime")
            available_transforms.append(base_name)
            
    if rand_selection:
        selection = np.random.choice(available_transforms,size=nc,replace=False)
    else:
        selection = np.arange(1,nc+1)
    
    for n in range(nc):
        #input_file = "Gen2l_64x32n1.lime"
        input_file = f"../gauge_confs/samples/{Nt}x32/{selection[n]}"
        gauge_file = f"../gauge_confs/transforms/{Nt}x32/{selection[n]}.gauge.lime"

        gauge_output = f"../gauge_confs/props/Nt{Nt}/{selection[n]}_t0_{t0}.prop"
        
        if os.path.exists(gauge_output):
            print("Cached file found. Loading...")
            prop = pd.read_csv(gauge_output).values[:]
            q = prop[:,:4]
            results = prop[:,4]
            D_results.append(results)       
        else:
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
            q = []
        
            # Loop over t,qx,qy,qz

            # Set lower bounds for Z3 averaging
            lqy = 0
            lqz = 0

            for t in [t0]:
                for qx in range(gf.shape[1]//4):
                    for qy in range(qx,gf.shape[2]//4):
                        for qz in range(qy,gf.shape[3]//4):
                            D_sum = 0
                            
                            # Fetch equivalent coordinates if Z3 averaging
                            if z3_avg:
                                z3_coords = unique_permute([qx,qy,qz])
                            else:
                                z3_coords = [[qx,qy,qz]]
                        
                            for coord in z3_coords:
                                for mu in [0,1,2,3]:

                                    # Decompose A in terms of Gell-Mann components
                                    A = gu.decompose_su3(gf.get_A((t,coord[0],coord[1],coord[2]),mu))
                                    A_neg = gu.decompose_su3(gf.get_A((-t,-coord[0],-coord[1],-coord[2]),mu))

                                    D_sum += np.dot(A,A_neg)

                            #results.append([[gf.get_qhat((t,qx,qy,qz),mu) for mu in [0,1,2,3]],D_sum])
                            q.append([t,qx,qy,qz])
                            results.append(D_sum/len(z3_coords))

                            if z3_avg:
                                lqz = qy
                        if z3_avg:
                            lqy = qx
        
            q = np.asarray(q)
            results = np.asarray(results)

            results_cleaned = []
            
            for pos,result in enumerate(results):
                if np.all(np.asarray(q)[pos] == 0):
                    results_cleaned.append(result*2/((gf.Nc**2-1)*gf.Nd*gf.V))
                else:
                    #results_cleaned.append([np.linalg.norm(q_improved(result[0])),result[1]*2/((gf.Nc**2-1)*(gf.Nd-1)*gf.V)])
                    results_cleaned.append(result*2/((gf.Nc**2-1)*(gf.Nd-1)*gf.V))

            results_cleaned = np.asarray(results_cleaned)
            D_results.append(results_cleaned)       
            print(f"{input_file} done.")
        
            # Save propagator values
            if save_prop:
                print("Saving propagator...")
                out_df = pd.DataFrame(np.hstack([q.real,results_cleaned.reshape(-1,1).real]))
            
                out_df.to_csv(gauge_output,index=None,header=['qt','qx','qy','qz','D(q)_s'])
        
    return q, D_results
        
if __name__ == "__main__":
    import sys
    spatial(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
