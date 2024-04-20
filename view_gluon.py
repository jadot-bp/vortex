#!/usr/bin/env python
# coding: utf-8

import lyncs_io as io
import numpy as np
import time
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as so
import scipy.stats as ss

import gluon_utils as gu
import gluon

import gvar as gv

__XI_R__ = 3.453  # Renormalised anisotropy
__XI_0__ = 4.3  # Bare gauge anisotropy
__NF__ = 3
__NC__ = 3

__dD__ = (39-4*__NF__)/(2*(33-2*__NF__))# See hep-lat/9809031 by T.W, D.L and J-I.S for detail

## Fitting functions ##

def logcorr(q,M,dD=__dD__):
    """The one-loop logarithmic correction to the propagator.
       See hep-lat/9809031 by T. W, D. L and J-I. S for detail."""
    
    return (0.5*np.log((q**2 + M**2)*(1/q**2 + 1/M**2)))**(-dD)

def Mq(q,M,L):
    """The logarithmic mass function defined in hep-lat/9809031
       to support the fit functions."""
    
    return M * (np.log((q**2 * 4*M**2)/L**2)/np.log(4*M**2/L**2))**(-6/11)

available_fits = ["gribov","stingl","marezoni","cornwall"]

class gribov:
    def fit(q,Z,M):
        """Gribov fit"""
        return Z * (q**2/(q**4 + M**4)) * logcorr(q,M)
    
    name = "Gribov"
    strform = "$D(q)=\dfrac{Zq^2}{q^4 + M^4}L(q,M)$"

class stingl:
    def fit(q,Z,A,M):
        """Stingl fit"""
        return Z * (q**2/(q**4 + 2 * A**2 * q**2 + M**4)) * logcorr(q,M)
    
    name = "Stingl"
    strform = "$D(q)=\dfrac{Zq^2}{q^4 + 2A^2q^2 + M^4}L(q,M)$"
    
class marezoni:
    def fit(q,Z,A,M):
        """Marezoni fit"""
        return Z/(q**(2+2*A) + M**2)
    
    name = "Marezoni"
    strform = "$D(q)=\dfrac{Z}{q^{2(1+\alpha)} + M^2}$"
    
class cornwall:
    def fit(q,Z,L,M):
        """Cornwall fit"""
        return Z/((q**2 * Mq(q,M,L)**2) * np.log((q**2 + 4*Mq(q,M,L)**2)/L**2))
    
    name = "Cornwall"
    strform = r"$Z\left[(q^2 + M^2(q^2)\mathrm{ln}\dfrac{q^2+4M^2(q^2)}{\Lambda^2})\right]^{-1}$"

## Utility functions ##

def chisq(ydata,yfit,yerr,ddof=1):
    ydata = np.asarray(ydata)
    yfit = np.asarray(yfit)
    yerr = np.asarray(yerr)
    
    return np.sum((ydata-yfit)**2 / yerr**2)/ddof

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
    """Return all cyclic permutations of coord."""
    
    items = list(itertools.permutations(coord))
   
    unique_items = []
    
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
            
    return np.asarray(unique_items)

def get_Z3_partners(coords):
    """Get all Z3 partners of a set of coords. Coords which are a
       partner of a previous coord return no partners."""
    
    z3_partners = [] # List of indices of Z3 partners
    z3_sig = [] # Mask (signature) of Z3 coordinates
    
    tmp_coords = np.copy(coords)
    
    while len(tmp_coords) > 0:
        # Get Z3 permutations of the coordinate
        z3_coords = unique_permute(tmp_coords[0])
        
        # Calculate the indices of the Z3 permutations in the temporary list
        z3_tmp_ids = np.argwhere([np.any(np.all(c_ == z3_coords, axis=1)) for c_ in tmp_coords]).flatten()
        
        # Calculate the indices of the Z3 permutations in the coordinate list
        z3_coord_ids = np.argwhere([np.any(np.all(c_ == z3_coords, axis=1)) for c_ in coords]).flatten()
        z3_partners.append(z3_coord_ids)
        
        # Store the index of the permuted (primary) coordinate
        z3_sig.append(z3_coord_ids[0])
        
        # Remove coordinate and permutations from temporary list and continue
        tmp_coords = np.delete(tmp_coords, z3_tmp_ids, axis=0)
    
    return z3_partners, z3_sig

def calculate_gz(p0,q,D,xi):
    """Calculate the renormalization correction factor g(z)."""
    p0_mask = q[:,0] == p0
    norm_p = np.asarray([np.linalg.norm(qp[1:]) for qp in q[p0_mask]])
    z = xi*p0/norm_p
    
    return (1+z**2)*D[p0_mask]/D[q[:,0] == 0]

def calculate_f(q,D,alpha,xi=1):
    """Calculate the spatial momentum component f(|p|), given by
       
       f(|p|) = 1/N_p0 \sum_p0 D(|p|,p0) (1+z^2)/g(z)
    
    """
    
    f = 0
    
    norm_q = np.asarray([np.linalg.norm(qi[1:]) for qi in q[q[:,0]==0]])
    
    for p0 in np.unique(q[:,0]):
        p0_mask = q[:,0] == p0
        
        z = xi*p0/norm_q[norm_q!=0]
        
        f += D[p0_mask][norm_q!=0] * (1+z**2)**(1-alpha)
    
    return f/len(np.unique(q[:,0]))

class propagator:
    def __init__(self,Nt,mode,n_samples='all',gtype="coulomb",path_to_props=None,compress=True,xi=1):
        """Fetch saved gluon propagator data"""
        
        self.Nt = Nt
        self.Ns = 32
        self.shape = np.asarray([self.Nt,self.Ns,self.Ns,self.Ns])
        self.gtype = gtype
        
        if path_to_props == None:
            raise Exception("Must specify path to propagator files! This is expected to be in the format:"\
                            "<path_to_props>/Nt<Nt>/<conf>.prop*")
        
        path_to_props += f"/Nt{Nt}/
        
        if gtype not in ['coulomb', 'landau']:
            raise Exception("Gauge type must be either 'coulomb' or 'landau'.")
        
        prop_names = []

        modes = {"VR":"-VR",
                 "VRS":"-VRS",
                 "VO":"-VO",
                 "VOS":"-VOS",
                 "FULL":"-UT"}
        
        if isinstance(mode, str) and mode in modes.keys():
            vmode = modes[mode]
        else:
            vmode = mode
        
        for prop in os.listdir(path_to_props):
            if gtype == 'coulomb' and prop.endswith(f'{vmode}.prop'):
                prop_names.append(prop)
            elif gtype == 'landau' and prop.endswith(f'{vmode}.prop.landau'):
                prop_names.append(prop)

        if len(prop_names) < 1:
            raise Exception(f"No propagator files found for gtype: {gtype}.")

        if isinstance(n_samples, list):
            suffix = ".landau" if gtype == "landau" else ""
            prop_names = [f"{i}.prop{suffix}" for i in n_samples]
        elif n_samples != 'all':
            prop_names = np.random.choice(prop_names, size=n_samples, replace=True)

        tmp_D = []
        tmp_D4 = []
        for prop in prop_names:
            data = pd.read_csv(f"{path_to_props}/{prop}").values[:]
            q_coord = data[:,:4]
            D_prop = data[:,4]
            tmp_D.append(D_prop)

            if gtype == "landau":
                D4_prop = data[:,5]
                tmp_D4.append(D4_prop)
        
        if compress and gtype == "landau":
            self.D = gv.gvar(np.mean(tmp_D,axis=0), np.std(tmp_D,axis=0))
            self.D4 = gv.gvar(np.mean(tmp_D4,axis=0), np.std(tmp_D4,axis=0))
            
        elif compress and gtype == "coulomb":
            self.D = gv.gvar(np.mean(tmp_D,axis=0), np.std(tmp_D,axis=0))
            
        else:
            self.D = np.asarray(tmp_D)
            if not gtype == "landau":
                self.D4 = np.asarray(tmp_D4)

        self.q = q_coord
        self.prop_info = prop_names

    def cone_cut(self, radius, q=None, D=None, D4=None, angle=np.pi/2, xi=1, cut_t=True,IR_cut=0,IR_radius=0):
        """Perform a cone cut along the BCD axis of the data."""

        if IR_radius == 0:
            IR_radius = radius
        
        if q == None:
            q = self.q
        if D == None:
            D = self.D
        if D4 == None and self.gtype == "landau":
            D4 = self.D4
        
        if q.shape[1] == 4 and (self.gtype == 'coulomb' or not cut_t):
            # Cone cut over each q_t slice

            cone_mask = []

            BCD_norm = np.ones(3)/np.linalg.norm(np.ones(3))

            N_qt = len(q[:,0] == np.unique(q[:,0])[0]) # Calculate number of q_t slices

            for coord in q[:N_qt,1:]:
                coord = get_qhat(coord,shape=(self.Ns,self.Ns,self.Ns))
                if np.all(coord == coord[0]): # Handle on-diagonal coordinates
                    r = 0
                    theta = 0
                else:
                    q_norm = np.linalg.norm(coord)
                    theta = np.arccos(np.dot(BCD_norm,coord)/q_norm)

                    r = q_norm * np.sin(theta)
                cone_mask.append(r <= radius and theta<angle)

            cone_mask *= len(q)//N_qt

            cone_mask = np.asarray(cone_mask, dtype=bool)

            self.q = q[cone_mask]
            self.D = D[cone_mask]
            
            return q[cone_mask], D[cone_mask]
        
        elif q.shape[1] == 4 and self.gtype == 'landau' and cut_t:

            cone_mask = []

            BCD_norm = np.ones(4)/np.linalg.norm(np.ones(4))

            for coord in q:
                
                coord = get_qhat(coord,shape=(self.Nt,self.Ns,self.Ns,self.Ns))
                coord *= np.asarray([xi,1,1,1]) # Correct for anisotropy
                
                if np.all(coord == coord[0]): # Handle on-diagonal coordinates
                    r = 0
                    theta = 0
                else:
                    q_norm = np.linalg.norm(coord)
                    theta = np.arccos(np.dot(BCD_norm,coord)/q_norm)

                    r = q_norm * np.sin(theta)
                    
                if np.linalg.norm(coord) <= IR_cut:
                    cone_mask.append(r <= IR_radius)
                else:
                    cone_mask.append(r <= radius and theta<angle)

            cone_mask = np.asarray(cone_mask, dtype=bool)

            self.q = q[cone_mask]
            self.D = D[cone_mask]
            self.D4 = D4[cone_mask]
            
            return q[cone_mask], D[cone_mask], D4[cone_mask]

        elif q.shape[1] == 3 and self.gtype == 'coulomb':
            cone_mask = []

            BCD_norm = np.ones(3)/np.linalg.norm(np.ones(3))

            for coord in q:
                coord = get_qhat(coord,shape=(self.Ns,self.Ns,self.Ns))
                if np.all(coord == coord[0]): # Handle on-diagonal coordinates
                    r = 0
                    theta=0
                else:
                    q_norm = np.linalg.norm(coord)
                    theta = np.arccos(np.dot(BCD_norm,coord)/q_norm)

                    r = q_norm * np.sin(theta)

                cone_mask.append(r <= radius and theta<angle)

            cone_mask = np.asarray(cone_mask, dtype=bool)

            self.q = q[cone_mask]
            self.D = D[cone_mask]
            
            return q[cone_mask], D[cone_mask]
        
        else:
            raise Exception("Could not perform cone cut.")

    def half_cut(self, q=None, D=None):
        """Cut the momenta half-way through the Brillouin zone."""
        
        if q == None:
            q = self.q
        if D == None:
            D = self.D
            
        cut_mask = np.ones(len(self.q))
        
        for i in range(4):
            cut_mask = np.logical_and(cut_mask, q[:,i] <= self.shape[i]//4)
        
        self.q = q[cut_mask]
        self.D = D[cut_mask]
        
        return q[cut_mask], D[cut_mask]
        
    def Z3_average(self, q=None, D=None, D4=None):
        """Average over the Z3-permuted coordinates."""

        if q is None:
            q = self.q
        if D is None:
            D = self.D
        if D4 is None and self.gtype == "landau":
            D4 = self.D4
        
        Nqs = int((self.Ns//2)**3) # Number of spatial points per timeslice

        if q.shape[1] == 4 and self.gtype == "landau":
            z3_partners, z3_sig = get_Z3_partners(q[:Nqs,1:])

            q_averaged = []
            D_averaged = []
            D4_averaged = []

            for t in range(len(q)//Nqs):
                q_averaged.extend([t,*coord] for coord in q[:,1:][z3_sig])
                D_averaged.extend([np.mean(D[t*Nqs:(t+1)*Nqs][z3_p]) for z3_p in z3_partners])
                D4_averaged.extend([np.mean(D4[t*Nqs:(t+1)*Nqs][z3_p]) for z3_p in z3_partners])
                
            return np.asarray(q_averaged,dtype=int), np.asarray(D_averaged), np.asarray(D4_averaged)
        
        if q.shape[1] == 4 and self.gtype == "coulomb":

            z3_partners, z3_sig = get_Z3_partners(q[:Nqs,1:])

            q_averaged = []
            D_averaged = []

            for t in range(len(q)//Nqs):
                q_averaged.extend([t,*coord] for coord in q[:,1:][z3_sig])
                D_averaged.extend([np.mean(D[t*Nqs:(t+1)*Nqs][z3_p]) for z3_p in z3_partners])
            return np.asarray(q_averaged,dtype=int), np.asarray(D_averaged)

        elif q.shape[1] == 3 and self.gtype == "coulomb":
            z3_partners, z3_sig = get_Z3_partners(q)

            q_averaged = q[z3_sig]
            D_averaged = [np.mean(D[z3_p]) for z3_p in z3_partners]

            return np.asarray(q_averaged,dtype=int), np.asarray(D_averaged)
        else:
            raise Exception("q must be either 4-dimensional (t,x,y,z) or 3-dimensional (x,y,z).")

    def norm_q(self,q=None):
        """Calculates the Fourier normed-q (q-hat) for the momentum."""
        
        if q is None:
            q = self.q
            
        if q.shape[1] == 4:
            shape=(self.Nt,self.Ns,self.Ns,self.Ns)
        else:
            shape=(self.Ns,self.Ns,self.Ns)
            
        q_normed = []
        
        for coord in q:
            q_normed.append(get_qhat(coord,shape))
            
        return np.asarray(q_normed)
            
    def correct_q(self,q=None,qtype="wilson"):
        """Applies the lattice correction to the momentum q."""
    
        if q is None:
            q = self.q
        
        assert qtype in ["wilson","improved"]

        correct_funcs = {"wilson": gluon.q_wilson,
                         "improved": gluon.q_improved}

        q_qtype = correct_funcs[qtype] # Correction function

        q_corrected = []

        for coord in q:
            q_corrected.append(q_qtype(coord))
            
        return np.asarray(q_corrected)
    
    def renormalize(self,xi=1):
        """Renormalizes the propagator."""
        
        if self.gtype == "landau":
            raise NotImplementedError("not implemented yet")
            
        elif self.gtype == "coulomb":
            
            # Calculate g(z)
            
            gz_fit = lambda x,a: np.asarray(x)*a
            
            norm_q = np.asarray([np.linalg.norm(qi[1:]) for qi in self.q[self.q[:,0] == 0]])
            
            zfactor = [] # 1+z^2
            gz = []
            gz_err = []
            
            # Iterate over e-slices
            for p0 in np.unique(self.q[:,0]):
                if p0 == 0:
                    continue # Avoid division by zero
                    
                p0_gz = calculate_gz(p0,self.q,self.D,xi)
                    
                z = xi*p0/norm_q[norm_q!=0]
                
                zfactor.extend(1+z**2)
                gz.extend([pgz.mean for pgz in p0_gz[norm_q!=0]])
                gz_err.extend([pgz.sdev for pgz in p0_gz[norm_q!=0]])
                
            popt, pcov = so.curve_fit(gz_fit, np.log(zfactor), np.log(gz), sigma=np.asarray(gz_err)/np.asarray(gz), absolute_sigma=True)
            
            self.alpha = gv.gvar(popt[0],np.sqrt(pcov[0][0]))
            
            try:
                self.chisq = chisq(np.log(gz), gz_fit(np.log(zfactor),self.alpha),np.asarray(gz_err)/np.asarray(gz),ddof=len(zfactor)-1)
            except:
                self.chisq = None
            
            f = calculate_f(self.q,self.D,self.alpha,xi)
            
            self.f = f
            
            self.gz = gv.gvar(gz,gz_err)
            self.zfactor = zfactor
