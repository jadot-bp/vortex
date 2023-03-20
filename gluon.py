#!/usr/bin/env python
# coding: utf-8

# In[2]:


import lyncs_io as io
import numpy as np
import time

import matplotlib.pyplot as plt

from scipy.fft import fftn

import gluon_utils as gu


def spatial(Nt,nc,t0):
    D_results = []

    for n in np.arange(1,nc+1):
        #input_file = "Gen2l_64x32n1.lime"
        input_file = f"../gauge_confs/{Nt}x32/Gen2l_{Nt}x32n{n}"
        gauge_file = f"../gauge_confs/transforms/Gen2l_{Nt}x32n{n}.gauge.lime"

        data = io.load(input_file, format="openqcd")
        gauge = io.load(gauge_file, format="lime")

        gf = gu.lattice(data,(Nt,32,4,3))
                
        # Apply gauge transformation
        gf.apply_gauge(gauge)

        # Transform to Fourier space
        gf.transform(axes=(1,2,3))

        results = []

        # Loop over t,qx,qy,qz
        start = time.time()
        for t in range(t0):
            for qx in range(gf.shape[1]//2):
                for qy in range(gf.shape[2]//2):
                    for qz in range(gf.shape[3]//2):
                        D_sum = 0

                        for mu in [1,2,3]:

                            # Decompose A in terms of Gell-Mann components
                            A = gu.decompose_su3(gf.get_A((t,qx,qy,qz),mu))
                            A_neg = gu.decompose_su3(gf.get_A((t,-qx,-qy,-qz),mu))

                            D_sum += np.sum(A*A_neg)

                        results.append([[gf.get_qhat((t,qx,qy,qz),mu) for mu in [0,1,2,3]],D_sum])
        end = time.time()

        results_cleaned = []

        def q_wilson(q_hat,a=1):
            return (2/a)*np.sin(np.asarray(q_hat)*a/2)
            #return np.asarray(q_hat)
            
        def q_improved(q_hat,a=1):
            return (2/a)*np.sqrt( np.sin(np.asarray(q_hat)*a/2)**2 + (1/3)*np.sin(np.asarray(q_hat)*a/2)**4 )

        for result in results:
            if np.all(result[0] == 0):
                results_cleaned.append([0,result[1]*2/((gf.Nc**2-1)*gf.Nd*gf.V)])
            else:
                results_cleaned.append([np.linalg.norm(q_improved(result[0])),result[1]*2/((gf.Nc**2-1)*(gf.Nd-1)*gf.V)])

        results_cleaned = np.asarray(results_cleaned)

        if n==1:
            D_results.append(results_cleaned[:,0])
        D_results.append(results_cleaned[:,1])
        print(f"{input_file} done.")

    return D_results
        
if __name__ == "__main__":
    import sys
    spatial(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
