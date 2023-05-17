#!/usr/bin/env python
# coding: utf-8

# In[2]:

import ctypes as c
import itertools
import os
import numpy as np
import numpy.ctypeslib as npc
import lyncs_io as io

import pandas as pd

import gluon_utils as gu


class c_double_complex(c.Structure):
    """Returns a double complex C structure."""

    _fields_ = [("real", c.c_double), ("imag", c.c_double)]

    @property
    def value(self):
        """Return the value of the structure."""

        return self.real+1j*self.imag


class GluonField(c.Structure):
    """Creates a struct to match gluon_field C structure."""

    _fields_ = [('Nt', c.c_int),
                ('Ns', c.c_int),
                ('Nd', c.c_int),
                ('Nc', c.c_int),
                ('U', c.POINTER(c_double_complex))]


def q_wilson(q_hat, a=1):
    """Calculate the momentum correction corresponding to the Wilson action."""

    return (2/a)*np.sin(np.asarray(q_hat)*a/2)

def q_improved(q_hat, a=1):
    """Calculate the momentum correction corresponding to the improved
    action."""

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
    
def py_spatial(Nt,Nc,t0,check_divA=False,calculate_z3=False,return_z3=False,rand_selection=True,save_prop=True, regenerate=True):
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
        selection = rand_selection
    
    for n in range(Nc):
        #input_file = "Gen2l_64x32n1.lime"
        input_file = f"../gauge_confs/samples/{Nt}x32/{selection[n]}"
        gauge_file = f"../gauge_confs/transforms/{Nt}x32/{selection[n]}.gauge.lime"

        gauge_output = f"../gauge_confs/props/Nt{Nt}/{selection[n]}.prop"
        z3_output = f"../gauge_confs/props/Nt{Nt}/{selection[n]}.prop.z3"
        
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
                print("div.A:",gf.py_evaluate_divA())
        
            # Transform to Fourier space
            gf.transform(axes=(0,1,2,3))

            results = []
            z3_results = []
            
            q = []
            z3_q = []
        
            # Loop over t,qx,qy,qz

            for t in range(gf.shape[0]//2):
                for qx in range(gf.shape[1]//2):
                    for qy in range(gf.shape[2]//2):
                        for qz in range(gf.shape[3]//2):
                            D = []
                            
                            # If Z3 averaging, calculate equivalent coordinates
                            if qz >= qy and qy >= qx and calculate_z3:
                                coords = unique_permute([qx,qy,qz])
                            else:
                                coords = [[qx,qy,qz]]

                            for coord in coords:
                                # Calculate propagator
                                
                                D_sum = 0
                                for mu in [1,2,3]:
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

def spatial(Nt, Nconf, check_divA=False, rand_selection=True, save_prop=True, regenerate=True):
    """Calculates the spatial gluon propagator using compiled code.

    Parameters:
        Nt: Temporal extent
        Nconf: Number of gauge configurations to sample

    Optional Parameters:
        check_divA [False]: Calculate the value of |div(A)|^2 for each configuration
        rand_selection [True]: Iterate through configurations randomly when sampling
        save_prop [True]: Save calculated values of the propagator
        
    Returns:
        q: Array of coordinates
        D_results: Array of propagator values
    """
    
    # Presets
    
    Ns = 32
    Nd = 4
    Nc = 3
    
    gauge_path = f"/home/ben/Work/gauge_confs/transforms"
    conf_path = f"/home/ben/Work/gauge_confs"
    prop_path = f"/home/ben/Work/gauge_confs/props"
    
    # Load gprop library
    
    script_dir = os.path.abspath(os.path.dirname(__file__))
    lib_path = os.path.join(script_dir, "libgprop.so")

    c.cdll.LoadLibrary(lib_path)

    LIB = c.CDLL(lib_path)

    LIB.calc_mom_space_scalarD.argtypes = [c.POINTER(GluonField),
                                           npc.ndpointer(np.complex128,
                                                         ndim=None,
                                                         flags="C_CONTIGUOUS")]
    LIB.calc_mom_space_scalarD.restypes = None

    D_results = []

    available_transforms = []
    
    for file in os.listdir(f"{gauge_path}/{Nt}x32"):
        if file.endswith(".gauge.lime"):
            base_name = file.rstrip(".gauge.lime")
            available_transforms.append(base_name)
            
    if isinstance(rand_selection,bool) and rand_selection:
        selection = np.random.choice(available_transforms,size=Nc,replace=False)
    else:
        selection = rand_selection
    
    # Check Nconf bound
    
    if Nconf > len(available_transforms):
        print(f"Defaulting to max Nconf={len(available_transforms)}")
        Nconf = len(available_transforms)
    
    # Generate q array:
    
    qt = np.arange(Nt//2) # Temporal coordinates
    qs = np.arange(Ns//2) # Spatial coordinates
    
    q = np.asarray(list(itertools.product(qt,qs,qs,qs)))
    
    for n in range(Nconf):
        input_file = f"{conf_path}/{Nt}x32/{selection[n]}"
        gauge_file = f"{gauge_path}/{Nt}x32/{selection[n]}.gauge.lime"

        prop_output = f"{prop_path}/Nt{Nt}/{selection[n]}.prop"
        z3_output = f"{prop_path}/Nt{Nt}/{selection[n]}.prop.z3"
        
        file_found = False
        
        if os.path.exists(prop_output) and not regenerate:
            print("Cached file found. Loading...")
            prop = pd.read_csv(prop_output).values[:]
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
            print(input_file)
            data = io.load(input_file, format="openqcd")
            gauge = io.load(gauge_file, format="lime")

            gf = gu.lattice(data,(Nt,Ns,Nd,Nc))
             
            # Apply gauge transformation
            gf.apply_gauge(gauge)

            if check_divA:
                # check divA
                print("div.A:", gf.py_evaluate_divA())
        
            # Transform to Fourier space
            gf.transform(axes=(0, 1, 2, 3))

            D_size = int((Nt/2)*(gf.Ns/2)**3)
            D = np.zeros(D_size, dtype='complex128') # Container for scalar propagator
            
            flattened = gf.lattice.flatten()
            
            gf_struct = GluonField()
            gf_struct.Nt = (c.c_int)(gf.Nt)
            gf_struct.Ns = (c.c_int)(gf.Ns)
            gf_struct.Nd = (c.c_int)(gf.Nd)
            gf_struct.Nc = (c.c_int)(gf.Nc)
            gf_struct.U = flattened.ctypes.data_as(c.POINTER(c_double_complex))

            LIB.calc_mom_space_scalarD(gf_struct, D)

            D_results.append(D.copy())
    
            # Save propagator values
            if save_prop:
                print("Saving propagator...")
                out_df = pd.DataFrame(np.hstack([q,D.reshape(-1,1).real]))
            
                out_df.to_csv(prop_output,index=None,header=['qt','qx','qy','qz','D(q)_s'])
    
    return q, D_results


if __name__ == "__main__":
    import sys
    spatial(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
