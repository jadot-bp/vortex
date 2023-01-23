#!/usr/bin/env python
# coding: utf-8

import lyncs_io as io
import numpy as np

import time

class gauge_fields:                                                             
    def __init__(self, lattice, params):                                        
        self.lattice = lattice                                                  
        self.Nt, self.Ns, self.Nd, self.Nc = params                             
                                                                                
    def get_fields(self):                                                       
        return self.lattice                                                     
                                                                                
    def get_link(self, coord, mu):                                              
        pos = *coord, mu                                                        
        return self.lattice[pos]
    
    def get_plaquette(self, coord, plane):                                      
        """ Calculates plaquette at coordinate in plane direction.              
                                                                                
        coord : coordinate in format (t,x,y,z)                                  
        plane : direction plaquette plane in format (mu, nu) (e.g. (1,2) for (x,y) plane)
                                                                                
        returns: plaquette matrix                                               
        """                                                                                                                                       
                                                                                
        mu, nu = plane                                                          
                                                                                
        # Get extent to handle periodic boundary conditions                     
        mu_extent = self.Ns if mu > 0 else self.Nt                              
        nu_extent = self.Ns if nu > 0 else self.Nt                              
                                                                                
        # Create direction vectors                                              
                                                                                
        mu_step = np.array([0,0,0,0])                                           
        mu_step[mu] = 1                                                         
                                                                                
        nu_step = np.array([0,0,0,0])                                           
        nu_step[nu] = 1                                                         
                                                                                
        P = 0                                                                   
                                                                                
        # get link in mu direction at coord                                     
        U_mu = self.get_link(coord,mu)                                          
                                                                                
        P = U_mu                                                                
                                                                                
        # get link in nu direction at coord + mu                                
        next_coord = np.asarray(coord) + mu_step                                
        next_coord[mu] %= mu_extent                                             
        U_nu = self.get_link(next_coord,nu)                                     
                                                                                
        P = np.matmul(P,U_nu)                                                   
                                                                                
       # get daggered link in mu direction at coord + nu                           
        next_coord = np.asarray(coord) + nu_step   
        next_coord[nu] %= nu_extent                                             
        U_mu_dag = np.conj(self.get_link(next_coord,mu)).T                      
                                                                                
        P = np.matmul(P,U_mu_dag)                                               
                                                                                
        # get daggered link in nu direction at coord                               
        next_coord = np.asarray(coord)                                          
        U_mu_nu_dag = np.conj(self.get_link(next_coord,nu)).T                   
                                                                                
        P = np.matmul(P,U_mu_nu_dag)                                            
                                                                                
        return P

# Read in openQCD gauge field file                                              
data = io.load("Gen2l_8x32n1",format='openqcd')                            
print("plaquette:",io.head("Gen2l_8x32n1",format="openqcd")['plaq'])                        
# Gauge files are in the shape (Nt,Ns,Ns,Ns,mu,color,color)

#unit_field = np.zeros_like(data)
#unit_field[:,:,:,:,:] = np.identity(3)
#data = unit_field

gf = gauge_fields(data, (8, 32, 4, 3))                                
                                                                                
start = time.time()                                                             
                                                                                
sumTrP = 0                                                                      
nP = 0                                                                          
n_sites = gf.Nt*gf.Ns**3                                    
                                                                                
#                tx        ty        tz        xy        xz        yz           
for plane in [(1,1,0,0),(1,0,1,0),(1,0,0,1),(0,1,1,0),(0,1,0,1),(0,0,1,1)]:     
    for t in range(gf.Nt):                                   
        for i in range(gf.Ns):                               
            for j in range(gf.Ns):                           
                for k in range(gf.Ns):                       
                                                                       
                    P = gf.get_plaquette((t,i,j,k),np.argwhere(plane).flatten())
                                                                                
                    sumTrP += np.trace(P).real                                  
                    nP += 1                                                     
                                                                    
end = time.time()                                                               
                                                                                
print(f"\nCalculated average plaquette in {end-start:.2f}s.")

calc_plaq = sumTrP/nP
file_plaq = io.head("Gen2l_8x32n1",format='openqcd')['plaq']
deviation = calc_plaq-file_plaq

print("File Value:\t\t",calc_plaq)
print("Calculated:\t",file_plaq)
print("Deviation:\t",abs(calc_plaq-file_plaq))
