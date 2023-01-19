import lyncs_io as io #https://github.com/Lyncs-API/lyncs
import numpy as np

import time

# Read in openQCD gauge field file
data = io.load("Gen2l_8x32n1.oqcd")

# Gauge files are in the shape (Nt,Ns,Ns,Ns,mu,color,color)

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
        mu, nu = plane

        mu_coord = np.array([0,0,0,0])
        mu_coord[mu] = 1

        nu_coord = np.array([0,0,0,0])
        nu_coord[nu] = 1

        P = 0

        U_mu = self.get_link(coord,mu)

        P = U_mu

        U_nu = self.get_link((np.asarray(coord)+mu_coord)%32,nu)

        P = np.matmul(P,U_nu)

        U_mu_dag = np.conj(self.get_link((np.asarray(coord) + mu_coord + nu_coord)%32,mu)).T

        P = np.matmul(P,U_mu_dag)

        U_mu_nu_dag = np.conj(self.get_link((np.asarray(coord) + nu_coord)%32,nu)).T

        P = np.matmul(P,U_mu_nu_dag)

        return P

gauge_fields = gauge_fields(data, (8, 32, 4, 3))

start = time.time()

sumTrP = 0
nP = 0
t = 0

# This block has some overcounting which will need correcting
for plane in [[1,2],[2,3],[3,1]]:
    for i in range(32):
        for j in range(32):
            for k in range(32):

                P = gauge_fields.get_plaquette((t,i,j,k),plane)

                sumTrP += np.trace(P)
                nP += 1

print(sumTrP/nP)
print(nP)

end = time.time()

print(f"\nCalculated average plaquette in {end-start:.2}s.")
