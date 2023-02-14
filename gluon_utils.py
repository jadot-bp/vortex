import lyncs_io as io
import numpy as np

class lattice:                                                             
    def __init__(self, lattice, params, is_conjugate=False):                                        
        self.lattice = lattice                                                  
        self.Nt, self.Ns, self.Nd, self.Nc = params        
        
        if self.Nd != 4: raise Exception("Dimensions other than 4 not yet supported!")
        if self.Nc != 3: raise Exception("Only SU(3) currently supported.")
        
        self.shape = tuple([self.Nt] + [self.Ns]*(self.Nd-1))
        self.V = np.prod(self.shape)
        
        self._is_conjugate = is_conjugate
    
        if np.prod(self.lattice.shape) != np.prod(self.shape)*self.Nc*self.Nc*self.Nd:
            raise Exception("Lattice shape mis-match!")
    
    
    def __repr__(self):
        return "lattice(lattice, params=(Nt,Ns,Nd,Nc), gauge_transform [optional])"
    
    
    def transform(self):
        """Performs the Fourier transform over the lattice."""
               
        self.lattice = np.fft.fftn(self.lattice,axes=(1,2,3))
        self._is_conjugate = not self._is_conjugate
        
    def step(self, pos, mu):
        """Calculates position one step in the mu direction from pos."""
        
        new_pos = np.asarray(pos).copy() ## Explicitly copy position array value
        dirn = 1
        
        if mu < 0:
            mu = abs(mu)
            dirn = -1
        
        new_pos[mu] += dirn
        new_pos[mu] %= self.shape[mu]  #Enforce periodic boundary conditions
        
        return new_pos
    
    def inverse_transform(self):
        """Performs the inverse Fourier transform over the lattice."""
               
        self.lattice = np.fft.ifftn(self.lattice,axes=(1,2,3))
        self._is_conjugate = not self._is_conjugate
    
    def apply_gauge(self,gauge,undo=False):
        """Applies SU(N) gauge transformation to the lattice."""
        
        assert gauge.shape == (self.Nt, self.Ns, self.Ns, self.Ns, self.Nc, self.Nc)
        
        assert not self._is_conjugate
        
        for t in range(self.Nt):
            for i in range(self.Ns):
                for j in range(self.Ns):
                    for k in range(self.Ns):
                        for mu in range(self.Nd):
                    
                            Gx = gauge[(t,i,j,k)]
                    
                            Ux = self.get_link((t,i,j,k),mu)
                    
                            xmu = self.step(np.array((t,i,j,k)),mu)
                    
                            Gxmu = gauge[tuple(xmu)]
                            
                            if undo:
                                self.lattice[t,i,j,k,mu] = np.matmul(np.conj(Gx.T),np.matmul(Ux,Gxmu))
                            else:
                                self.lattice[t,i,j,k,mu] = np.matmul(Gx,np.matmul(Ux,np.conj(Gxmu.T)))
        
    def get_lattice(self):
        """Fetches the stored lattice."""
        return self.lattice                                                     
                   
        
    def get_link(self, coord, mu):
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
        if self._is_conjugate:
            raise Exception("Plaquette calculation not yet supported for conjugate lattices.")
                                                                                
        mu, nu = plane                                                          
                                                                               
        P = 0                                                                   
                                                                                
        # get link in mu direction at coord                                     
        U_mu = self.get_link(coord,mu)                                          
                                                                                
        P = U_mu                                                                
                                                                                
        # get link in nu direction at coord + mu                                                                          
        U_nu = self.get_link(self.step(coord,mu),nu)                                     
                                                                                
        P = np.matmul(P,U_nu)                                                   
                                                                                
        # get daggered link in mu direction at coord + nu                                           
        U_mu_dag = np.conj(self.get_link(self.step(coord,nu),mu)).T                      
                                                                                
        P = np.matmul(P,U_mu_dag)                                               
                                                                                
        # get daggered link in nu direction at coord                                      
        U_mu_nu_dag = np.conj(self.get_link(coord,nu)).T                   
                                                                                
        P = np.matmul(P,U_mu_nu_dag)                                            
                                                                                
        return P
    
    def evaluate_average_plaquette(self):
        
        tP = 0 # temporal plaquette
        
        #                 tx        ty        tz
        for plane in [(1,1,0,0),(1,0,1,0),(1,0,0,1)]:
            for t in range(self.Nt):
                for i in range(self.Ns):
                    for j in range(self.Ns):
                        for k in range(self.Ns):
                            
                            tP += np.trace(self.get_plaquette((t,i,j,k),np.argwhere(plane).flatten())).real
        
        sP = 0 # spatial plaquette
        
        #                 xy        xz        yz
        for plane in [(0,1,1,0),(0,1,0,1),(0,0,1,1)]:
            for t in range(self.Nt):
                for i in range(self.Ns):
                    for j in range(self.Ns):
                        for k in range(self.Ns):
                            
                            sP += np.trace(self.get_plaquette((t,i,j,k),np.argwhere(plane).flatten())).real            
        
        return tP/(3*np.prod(self.shape)), sP/(3*np.prod(self.shape))
    
    def evaluate_divA(self,pattern="coulomb"):
        """Evaluates the gauge fixing condition div.A=0 over the lattice."""
        
        assert pattern in ["coulomb", "laplace"]
        
        if pattern == "laplace":
            raise Exception("Laplace gauge condition not yet supported.")
        if self._is_conjugate == True:
            raise Exception("Conjugate lattice divergence not yet supported.")
    
        divA2 = 0       
        for t in range(self.Nt):
            for i in range(self.Ns):
                for j in range(self.Ns):
                    for k in range(self.Ns):
                        
                        divA = 0
                        for mu in [1,2,3]:
                            x = np.array([t,i,j,k])
                            
                            A_backward = self.get_A(self.step(x,-mu),mu)
                            A = self.get_A(x,mu)
                            
                            divA += A-A_backward
                            
                        divA2 += np.sum(decompose_su3(divA)**2)
                            
        return divA2/np.prod(self.shape)
    
    def get_qhat(self,coord,mu,a=1):
        """Calculates q^_\mu corresponding to coord."""
        
        coord = np.asarray(coord)
        
        return (2*np.pi/a)*(coord[mu]/self.shape[mu])
    
    def get_A(self, coord, mu, a=1):
        
        coord = np.asarray(coord)
        
        if not self._is_conjugate:
            
            # Fetch A = (U-U^\dag)_traceless/2i
            U = self.get_link(coord,mu)
            
            U_Udag = np.zeros((3,3),dtype='complex128')
            
            tr = 0
            for i in range(3):
                tr += 2j*U[i,i].imag
                U_Udag[i,i] = 2j*U[i,i].imag
                
            tr /= self.Ns
            
            for i in range(3):
                U_Udag[i,i] -= tr
                
            for i in range(3):
                for j in range(3):
                    if i!=j:
                        U_Udag[i,j] = U[i,j]-np.conj(U[j,i])
                        U_Udag[j,i] = U[j,i]-np.conj(U[i,j])
            
            #return (U-np.conj(U.T) - 2j*np.trace(U)/(self.Nc))/2j
            return -U_Udag/2j
        
        else:
        
            U = self.get_link(coord,mu)
            U_neg = self.get_link(-coord,mu)
            B = U - np.conj(U_neg.T)
        
            return np.exp(-0.5j*a*self.get_qhat(coord, mu))/(2j*a) * (B - np.trace(B)/self.Nc)
            
    def evaluate_e2(self):
        """Calculates the gauge-fixing proxy e2 from Jesuel Marques' gauge-fixing code."""
        
        e2 = 0
        
        for t in range(self.Nt):
            for i in range(self.Ns):
                for j in range(self.Ns):
                    for k in range(self.Ns):
                        
                        divA = 0
                        for mu in [1,2,3]:
                            x = np.array([t,i,j,k])
                            
                            A_backward = self.get_A(self.step(x,-mu),mu)
                            A = self.get_A(x,mu)
                            
                            divA += (A-A_backward)
                           
                        divA_components = decompose_su3(divA)
                            
                        e2 += np.sum(np.asarray(divA_components)**2)
        
        return e2/np.prod(self.shape)
        
        
def gell_mann(number=None):
    
    if number != None:
        assert isinstance(number,int)
        assert number >= 0 and number < 8
    
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
    
    g8 = 1/np.sqrt(3) * np.asarray(([1,0,0],
                                    [0,1,0],
                                    [0,0,-2]))
    
    if number == None:
        return (g1, g2, g3, g4, g5, g6, g7, g8)
    else:
        return (g1, g2, g3, g4, g5, g6, g7, g8)[number]

def decompose_su3(matrix):
    """Decomposes an SU(3) matrix into its color components in the style of Jesuel Marques' code."""
    
    m1 = matrix[0,1].real + matrix[1,0].real
    m2 = -matrix[0,1].imag + matrix[1,0].imag
    m3 = matrix[0,0] - matrix[1,1]
    m4 = matrix[0,2].real + matrix[2,0].real
    m5 = -matrix[0,2].imag + matrix[2,0].imag
    m6 = matrix[1,2].real + matrix[2,1].real
    m7 = -matrix[1,2].imag + matrix[2,1].imag
    m8 = (matrix[0,0].real + matrix[1,1].real -2.0*matrix[2,2].real)/np.sqrt(3)
    
    return np.array((m1,m2,m3,m4,m5,m6,m7,m8))
    
    
        
    
    