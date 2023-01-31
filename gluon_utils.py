import lyncs_io as io
import numpy as np

class lattice:                                                             
    def __init__(self, lattice, params, gauge_transform = None):                                        
        self.lattice = lattice                                                  
        self.Nt, self.Ns, self.Nd, self.Nc = params        
        self.gauge_transform = gauge_transform
        
        self.shape = (self.Nt, self.Ns, self.Ns, self.Ns)
    
    def __repr__(self):
        return "lattice(lattice, params=(Nt,Ns,Nd,Nc), gauge_transform [optional])"
    
    def get_lattice(self):                                                       
        return self.lattice                                                     
                                                                                
    def get_link(self, coord, mu, apply_transform = False):                                              
        pos = *coord, mu
        
        if apply_transform == True:
            if not isinstance(self.gauge_transform,np.ndarray):
                raise Exception("Gauge transformation field not specified.")
                
            gx = self.gauge_transform[tuple(coord)]

            coord_mu = list(coord)
            coord_mu[mu] += 1
            coord_mu[mu] %= self.shape[mu]
            gxmu = self.gauge_transform[tuple(coord_mu)]
            return np.matmul(gx,np.matmul(self.lattice[pos],np.conj(gxmu).T))
        else:
            return self.lattice[pos]
    
    def get_plaquette(self, coord, plane,apply_transform = False):                                      
        """ Calculates plaquette at coordinate in plane direction.              
                                                                                
        coord : coordinate in format (t,x,y,z)                                  
        plane : direction plaquette plane in format (mu, nu) (e.g. (1,2) for (x,y) plane)
                                                                                
        returns: plaquette matrix                                               
        """                                                                                                                                       
                                                                                
        mu, nu = plane                                                          
                                                                                
        # Get extent to handle periodic boundary conditions                     
        mu_extent = self.shape[mu]                             
        nu_extent = self.shape[nu]                             
                                                                                
        # Create direction vectors                                              
                                                                                
        mu_step = np.array([0,0,0,0])                                           
        mu_step[mu] = 1                                                         
                                                                                
        nu_step = np.array([0,0,0,0])                                           
        nu_step[nu] = 1                                                         
                                                                                
        P = 0                                                                   
                                                                                
        # get link in mu direction at coord                                     
        U_mu = self.get_link(coord,mu,apply_transform)                                          
                                                                                
        P = U_mu                                                                
                                                                                
        # get link in nu direction at coord + mu                                
        next_coord = np.asarray(coord) + mu_step                                
        next_coord[mu] %= mu_extent                                             
        U_nu = self.get_link(next_coord,nu,apply_transform)                                     
                                                                                
        P = np.matmul(P,U_nu)                                                   
                                                                                
       # get daggered link in mu direction at coord + nu                           
        next_coord = np.asarray(coord) + nu_step   
        next_coord[nu] %= nu_extent                                             
        U_mu_dag = np.conj(self.get_link(next_coord,mu,apply_transform)).T                      
                                                                                
        P = np.matmul(P,U_mu_dag)                                               
                                                                                
        # get daggered link in nu direction at coord                               
        next_coord = np.asarray(coord)                                          
        U_mu_nu_dag = np.conj(self.get_link(next_coord,nu,apply_transform)).T                   
                                                                                
        P = np.matmul(P,U_mu_nu_dag)                                            
                                                                                
        return P