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

def spatial(Nt, Nconf, mode, xi=1.0, check_divA=False, rand_selection=True, save_prop=True, regenerate=True, pattern='coulomb',transform=True):
    """Calculates the spatial gluon propagator using compiled code.

    Parameters:
        Nt: Temporal extent
        Nconf: Number of gauge configurations to sample
        mode: Vortex mode ("VR","VO" or "VOS".)

    Optional Parameters:
        xi [1.0]: The lattice anisotropy a_s/a_t
        check_divA [False]: Calculate the value of |div(A)|^2 for each configuration
        rand_selection [True]: Iterate through configurations randomly when sampling
        save_prop [True]: Save calculated values of the propagator
        regenerate [True]: Ignore saved propagators and regenerate
        pattern ['Coulomb']: Gauge fixing
        
    Returns:
        q: Array of coordinates
        D_results: Array of propagator values
    """
    
    # Presets
    
    NS = 32
    ND = 4
    NC = 3

    # End Presets
    
    modes = {"VO":"vortex-only",
             "VOS":"vortex-only",
             "VR":"vortex-removed",
             "VRS":"vortex-removed",
             "UT":"full"}
    
    if isinstance(mode,str) and mode in modes.keys():
        vmode = modes[mode]
    else:
        mode,vmode = list(mode.items())[0]
        
    if pattern == 'landau':
        gauge_path = f"/home/ben/Work/gauge_confs/transforms/landau/{vmode}"
        MU_START = 0
    else:
        gauge_path = f"/home/ben/Work/gauge_confs/transforms/{vmode}"
        MU_START = 1    
        
    conf_path = f"/home/ben/Work/gauge_confs/confs/{vmode}"
    prop_path = f"/home/ben/Work/gauge_confs/props"
    
    # Load gprop library
    
    script_dir = os.path.abspath(os.path.dirname(__file__))
    lib_path = os.path.join(script_dir, "libgprop.so")

    c.cdll.LoadLibrary(lib_path)

    LIB = c.CDLL(lib_path)

    LIB.calc_mom_space_scalarD.argtypes = [c.POINTER(GluonField),
                                           npc.ndpointer(np.complex128,
                                                         ndim=None,
                                                         flags="C_CONTIGUOUS"),
                                           npc.ndpointer(np.complex128,
                                                         ndim=None,
                                                         flags="C_CONTIGUOUS"),
                                           c.c_int,
                                           c.c_float]
    
    LIB.calc_mom_space_scalarD.restypes = None

    D_results = []
    D4_results = []

    available_transforms = []

    if transform == True:
    
        for file in os.listdir(f"{gauge_path}/{Nt}x32"):
            if file.endswith(".gauge.lime"):
                base_name = file.rstrip(".gauge.lime")
                available_transforms.append(base_name)
            
    if isinstance(rand_selection,bool) and rand_selection and transform:
        selection = np.random.choice(available_transforms,size=Nconf,replace=False)
    else:
        selection = rand_selection
    
        # Check Nconf bound
    
    if Nconf > len(available_transforms) and transform:
        print(f"Defaulting to max Nconf={len(available_transforms)}")
        Nconf = len(available_transforms)
    
    # Generate q array:
    
    qt = np.arange(Nt//2) # Temporal coordinates
    qs = np.arange(NS//2) # Spatial coordinates
    
    q = np.asarray(list(itertools.product(qt,qs,qs,qs)))
    
    for n in range(Nconf):
        input_file = f"{conf_path}/{Nt}x32/{selection[n]}"
        
        if transform == True:
            gauge_file = f"{gauge_path}/{Nt}x32/{selection[n]}.gauge.lime"

        prop_output = f"{prop_path}/Nt{Nt}/{selection[n]}{f'-UT' if mode == 'UT' else ''}.prop{'.landau' if pattern == 'landau' else ''}"
        
        file_found = False
        
        if os.path.exists(prop_output) and not regenerate:
            print("Cached file found. Loading...")
            prop = pd.read_csv(prop_output).values[:]
            q = prop[:,:4]
            results = prop[:,4]
            D_results.append(results)
            file_found = True
        
        if not file_found:
            print("No cached file found. Generating...")
            print(input_file, gauge_file)
            
            data = io.load(input_file, format="openqcd")    

            gf = gu.lattice(data,(Nt,NS,ND,NC))
            
            
            if transform == True:
                #  Load gauge transform
                gauge = io.load(gauge_file, format="lime")
                
                # Apply gauge transformation
                gf.apply_gauge(gauge)

            if check_divA:
                # check divA
                print("div.A:", gf.evaluate_divA(pattern=pattern, xi=xi))
        
            # Transform to Fourier space
            gf.transform(axes=(0, 1, 2, 3))

            D_size = int((Nt/2)*(gf.Ns/2)**3)
            D = np.zeros(D_size, dtype='complex128') # Container for scalar propagator
            D4 = np.zeros(D_size, dtype='complex128')
            
            flattened = gf.lattice.flatten()
            
            gf_struct = GluonField()
            gf_struct.Nt = (c.c_int)(gf.Nt)
            gf_struct.Ns = (c.c_int)(gf.Ns)
            gf_struct.Nd = (c.c_int)(gf.Nd)
            gf_struct.Nc = (c.c_int)(gf.Nc)
            gf_struct.U = flattened.ctypes.data_as(c.POINTER(c_double_complex))

            LIB.calc_mom_space_scalarD(gf_struct, D, D4, MU_START, xi)

            D_results.append(D.copy())
            if pattern == 'landau':
                D4_results.append(D4.copy())
    
            # Save propagator values
            if save_prop:
                print("Saving propagator...")
                if pattern == "landau":
                    out_df = pd.DataFrame(np.hstack([q,D.reshape(-1,1).real,D4.reshape(-1,1).real]))    
                    out_df.to_csv(prop_output,index=None,header=['qt','qx','qy','qz','D(q)_s','D4(q)_s'])
                else:
                    out_df = pd.DataFrame(np.hstack([q,D.reshape(-1,1).real]))    
                    out_df.to_csv(prop_output,index=None,header=['qt','qx','qy','qz','D(q)_s'])
    if pattern == 'landau':
        return q, D_results, D4_results
    else:
        return q, D_results


if __name__ == "__main__":
    import sys
    spatial(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
