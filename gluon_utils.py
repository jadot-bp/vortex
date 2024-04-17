import lyncs_io as io
import numpy as np

import ctypes as c
import numpy.ctypeslib as npc

import os

class lattice:                                                             
    def __init__(self, lattice, params, is_conjugate=False):                                        
        self.lattice = lattice.astype("complex128")
        self.Nt, self.Ns, self.Nd, self.Nc = params        
        
        
        # Assert dimension checks
        if self.Nd != 4: raise Exception("Dimensions other than 4 not yet supported!")
        if self.Nc != 3: raise Exception("Only SU(3) currently supported.")
        
        # Generate geometric parameters
        self.shape = tuple([self.Nt] + [self.Ns]*(self.Nd-1))
        self.V = np.prod(self.shape)
        
        if np.prod(self.lattice.shape) != np.prod(self.shape)*self.Nc*self.Nc*self.Nd:
            raise Exception("Lattice shape mis-match!")
        
        # Axis conjugation for Fourier-transformed lattices
        
        if isinstance(is_conjugate,bool):
            self._is_conjugate = [is_conjugate]*self.Nd
        elif isintance(is_conjugate,(tuple,list)):
            self._is_conjugate = is_conjugate
        else:
            raise Exception("Unknown is_conjugate specified.")
    
    def __repr__(self):
        return "lattice(lattice, params=(Nt,Ns,Nd,Nc), is_conjugate[=False])"
    
    
    def transform(self, axes=(0,1,2,3)):
        """Performs the Fourier transform over the lattice."""
               
        self.lattice = np.fft.fftn(self.lattice,axes=axes)
        
        for ax in axes:
            self._is_conjugate[ax] = not self._is_conjugate[ax]
        
    def step(self, pos, mu, dirn=1):
        """Calculates position one step in the mu direction from pos."""
        
        new_pos = np.asarray(pos).copy() ## Explicitly copy position array value
                
        new_pos[mu] += dirn
        new_pos[mu] %= self.shape[mu]  #Enforce periodic boundary conditions
        
        return new_pos
    
    def inverse_transform(self, axes=(1,2,3)):
        """Performs the inverse Fourier transform over the lattice."""
               
        self.lattice = np.fft.ifftn(self.lattice,axes=axes)
        
        for ax in axes:
            self._is_conjugate[ax] = not self._is_conjugate[ax]
    
    def py_apply_gauge(self,gauge,undo=False):
        """Applies SU(N) gauge transformation to the lattice."""
        
        assert gauge.shape == (self.Nt, self.Ns, self.Ns, self.Ns, self.Nc, self.Nc)
        
        assert not np.any(self._is_conjugate)
        
        for t in range(self.Nt):
            for i in range(self.Ns):
                for j in range(self.Ns):
                    for k in range(self.Ns):
                        for mu in range(self.Nd):
                    
                            Gx = gauge[(t,i,j,k)]
                    
                            Ux = self.get_link((t,i,j,k),mu)
                    
                            xmu = self.step(np.array((t,i,j,k)),mu)
                    
                            Gxmu = gauge[tuple(xmu)]
                                                        
                            if undo == True:
                                self.lattice[t,i,j,k,mu] = np.matmul(np.conj(Gx.T),np.matmul(Ux,Gxmu))
                            else:
                                self.lattice[t,i,j,k,mu] = np.matmul(Gx,np.matmul(Ux,np.conj(Gxmu.T)))
                                
    def apply_gauge(self,gauge):
        """Applies SU(N) gauge transformation to the lattice."""
    
        script_dir = os.path.abspath(os.path.dirname(__file__))
        lib_path = os.path.join(script_dir, "libgutils.so")
        
        c.cdll.LoadLibrary(lib_path)
    
        LIB = c.CDLL(lib_path)
    
        LIB.gauge_transform.argtypes = [npc.ndpointer(np.complex128, ndim=None, flags="C_CONTIGUOUS"),
                                        npc.ndpointer(np.complex128, ndim=None, flags="C_CONTIGUOUS"),
                                        c.c_int,
                                        c.c_int,
                                        c.c_int,
                                        c.c_int]
        
        LIB.gauge_transform.restype = None
    
        LIB.gauge_transform(self.lattice,gauge.astype('complex128'),self.Nt,self.Ns,self.Nd,self.Nc)
        
        return 0
    
    def get_lattice(self):
        """Fetches the stored lattice."""
        return self.lattice                                                     
                   
        
    def get_link(self, coord, mu):
        """Fetches the link field in the mu-direction at coord."""
        
        coord = np.asarray(coord)
                
        pos = np.array(([*coord, mu]))
        
        return self.lattice[tuple(pos)]
    
    
    def get_plaquette(self, coord, plane):                                      
        """ Calculates plaquette at coord in plane direction.              
                                                                                
        coord : coordinate in format (t,x,y,z)                                  
        plane : direction plaquette plane in format (mu, nu) (e.g. (1,2) for (x,y) plane)
                                                                                
        returns: plaquette matrix                                               
        """                                                                                                                             
        if np.any(self._is_conjugate):
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
    
    def py_evaluate_divA(self,pattern="coulomb",T_INDX=0):
        """Evaluates the gauge fixing condition div.A=0 over the lattice."""
        
        assert pattern in ["coulomb", "landau"]
        
        if pattern == "landau":
            mu_coords = [0,1,2,3]
            xi = 1/(3.444*4.3) # Lattice anisotropy multiplied by bare gauge anisotropy
            
        elif pattern == "coulomb":
            mu_coords = [1,2,3]
            xi = 1.0
            
        if np.any(self._is_conjugate) == True:
            raise Exception("Conjugate lattice divergence not yet supported.")
    
        divA2 = 0       
        for t in range(self.Nt):
            for i in range(self.Ns):
                for j in range(self.Ns):
                    for k in range(self.Ns):      
                        divA = 0
                        for mu in mu_coords:
                            x = np.array([t,i,j,k])
                            
                            A_backward = self.get_A(self.step(x,mu,dirn=-1),mu)
                            A = self.get_A(x,mu)
                            
                            if mu == T_INDX:
                                divA += xi*(A-A_backward)
                            else:
                                divA += A-A_backward
                            
                        divA2 += np.sum(decompose_su3(divA)**2)
                            
        return divA2/self.V
    
    def evaluate_divA(self, pattern="coulomb",xi=None):
        """Evaluates the div.A condition over the lattice."""
    
        assert pattern in ["coulomb", "landau"]
        
        if pattern == "coulomb":
            XI = 1.0
            MU_START = 1
        elif pattern == "landau":
            MU_START = 0
            XI = 3.453**2 # Lattice anisotropy
    
        if pattern == "landau" and xi != None:
            XI = xi*xi
    
        script_dir = os.path.abspath(os.path.dirname(__file__))
        lib_path = os.path.join(script_dir, "libgutils.so")
        
        c.cdll.LoadLibrary(lib_path)
    
        LIB = c.CDLL(lib_path)
    
        LIB.evaluate_divA.argtypes = [npc.ndpointer(np.complex128, ndim=None, flags="C_CONTIGUOUS"),
                                      c.c_int,
                                      c.c_int,
                                      c.c_int,
                                      c.c_int,
                                      c.c_int,
                                      c.c_double]
        
        LIB.evaluate_divA.restype = c.c_double
    
        divA = LIB.evaluate_divA(self.lattice,self.Nt,self.Ns,self.Nd,self.Nc, MU_START, XI)
        
        return divA
    
    def py_evaluate_F(self,pattern="coulomb",T_INDX=0):
        """Evaluates the gauge fixing condition F[U] over the lattice."""
        
        assert pattern in ["coulomb", "landau"]
        
        if pattern == "landau":
            mu_coords = [0,1,2,3]
            xi = 3.453**2 # Lattice anisotropy multiplied by bare gauge anisotropy
            
        elif pattern == "coulomb":
            mu_coords = [1,2,3]
            xi = 1.0
            
        if np.any(self._is_conjugate) == True:
            raise Exception("Conjugate lattice divergence not yet supported.")
    
        F = 0       
        for t in range(self.Nt):
            for i in range(self.Ns):
                for j in range(self.Ns):
                    for k in range(self.Ns):      
                        for mu in mu_coords:
                            x = np.array([t,i,j,k])
                            
                            U = self.get_link(x,mu)
                            
                            if mu == T_INDX:
                                F += xi*np.real(np.trace(U))
                            else:
                                F += np.real(np.trace(U))
                            
        return F/(self.V*self.Nc*(self.Nd-1))
    
    def get_qhat(self,coord,mu,a=1):
        """Calculates q^_\mu corresponding to coord."""
        
        coord = np.asarray(coord)
        
        return (2*np.pi/a)*(coord[mu]/self.shape[mu])
    
    def get_A(self, coord, mu, a=1, g0=1):
        
        coord = np.asarray(coord)
        
        if not self._is_conjugate[mu]:
            
            # Fetch A = (U-U^\dag)_traceless/2i
            U = self.get_link(coord,mu)
            
            return tracelessHermConjSubtraction(U,Nc=self.Nc)/2j
        
        else:
        
            U = self.get_link(coord,mu)
            U_neg = self.get_link(-coord,mu)
        
            B_traceless = tracelessHermConjSubtraction(U,Udag=np.conj(U_neg.T), Nc=self.Nc)
        
            return np.exp(-0.5j*a*self.get_qhat(coord, mu))/(2j*a*g0) * B_traceless
            
        
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
    
def tracelessHermConjSubtraction(U,Udag=None,Nc=3):
    """Calculates the traceless Hermitian conjugate subtraction of U: {U-U^\dag}-tr."""   
    
    if np.shape(Udag) != (Nc, Nc):
        if Udag == None:
            Udag = np.conj(U.T)
        else:
            raise Exception('Udag (if specified) must be a matrix of shape (Nc,Nc).')
    
    U_Udag_trcless = (U - Udag) - np.trace(U - Udag)*np.identity(Nc)/Nc # Confirmed correct format by divA tests
                
    return U_Udag_trcless
