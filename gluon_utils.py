import lyncs_io as io
import numpy as np

class lattice:                                                             
    def __init__(self, lattice, params, is_conjugate=False):                                        
        self.lattice = lattice                                                  
        self.Nt, self.Ns, self.Nd, self.Nc = params        
        
        self.shape = tuple([self.Nt] + [self.Ns]*(self.Nd-1))
        self.V = np.prod(self.shape)
        self._is_conjugate = is_conjugate
    
    
    def __repr__(self):
        return "lattice(lattice, params=(Nt,Ns,Nd,Nc), gauge_transform [optional])"
    
    
    def transform(self):
        """Performs the Fourier transform over the lattice."""
               
        self.lattice = np.fft.fftn(self.lattice,axes=(0,1,2,3))
        self._is_conjugate = not self._is_conjugate
        
        
    def inverse_transform(self):
        """Performs the inverse Fourier transform over the lattice."""
               
        self.lattice = np.fft.ifftn(self.lattice,axes=(0,1,2,3))
        self._is_conjugate = not self._is_conjugate
    
    
    def get_lattice(self):
        """Fetches the stored lattice."""
        return self.lattice                                                     
                   
        
    def get_link(self, coord, mu, ):
        """Fetches the link field in the mu-direction at coord."""
        
        coord = np.asarray(coord)
        
        if np.any(coord < 0) and not self._is_conjugate:
            raise IndexError("Negative coordinates not supported for non-conjugated lattices.")
        if np.any(2*coord/np.asarray(self.shape) > 1) and self._is_conjugate:
            raise IndexError("Coordinates greater in magnitude than half the lattice extent invalid for conjugated lattices.")
        
        pos = np.array(([*coord, mu]))
        
        return self.lattice[tuple(pos)]
    
    
    def get_plaquette(self, coord, plane):                                      
        """ Calculates plaquette at coord in plane direction.              
                                                                                
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
    
    def evaluate_average_plaquette():
        ## TO DO
        return None
    
    def get_qhat(self,coord,mu,a=1):
        """Calculates q^_\mu corresponding to coord."""
        
        coord = np.asarray(coord)
        
        return (2*np.pi/a)*(coord[mu]/self.shape[mu])
    
    def get_A(self, coord, mu, a=1):
        
        coord = np.asarray(coord)
        
        if not self._is_conjugate:
            # Fetch A = (U-U^\dag)_traceless/2i
            U = self.get_link(coord,mu)
            
            return (U-np.conj(U.T))/2j - np.trace(U-np.conj(U.T))/(2j*self.Nc)
        
        else:
        
            U = self.get_link(coord,mu)
            U_neg = self.get_link(-coord,mu)
            B = U - np.conj(U_neg.T)
        
            return np.exp(-0.5j*a*self.get_qhat(coord, mu))/(2j*a) * (B - np.trace(B)/self.Nc)
            
            
        
def gell_mann(number=None):
    
    if number != None:
        assert isinstance(number,int)
        assert number > 0 and number < 9
    
    g1 = np.asarray(([0,1,0],
                     [1,0,0],
                     [0,0,0]))
    
    g2 = np.asarray(([0,-1j,0],
                     [1j,0,0],
                     [0,0,0]))
    
    g3 = np.asarray(([1,0,0],
                     [0,-1,0],
                     [0,0,0]))
    
    g4 = np.asarray(([0,0,1],
                     [0,0,0],
                     [1,0,0]))
    
    g5 = np.asarray(([0,0,-1j],
                     [0,0,0],
                     [1j,0,0]))
    
    g6 = np.asarray(([0,0,0],
                     [0,0,1],
                     [0,1,0]))
    
    g7 = np.asarray(([0,0,0],
                     [0,0,-1j],
                     [0,1j,0]))
    
    g8 = 1/np.sqrt(3) * np.asarray(([1,0,1],
                                    [0,1,0],
                                    [0,0,-2]))
    
    if number == None:
        return (g1, g2, g3, g4, g5, g6, g7, g8)
    else:
        return (g1, g2, g3, g4, g5, g6, g7, g8)[number]