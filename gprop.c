#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>

#define DIM 7
#define MU_START 0
#define NUM_THREADS 5

int _calc_idx(int pos[], int shape[], int Nd){
    // Calculate the row-major address in  memory at position pos for given shape.
    //
    // Args:
    //
    // pos[]   : Position within the array
    // shape[] : Shape (dimensionality) of the array
    // Nd      : Number of dimensions of the array.
        
    int idx = 0;
    
    for(int k=0; k < Nd; k++){
        int prod_dim = 1;
        
        for(int l=k+1; l < Nd; l++){
            prod_dim *= shape[l];
        }
        idx += prod_dim*pos[k];
    }
    return idx;
}

int *decompose_su3(double complex M[3][3], double complex components[8]){
    
    components[0] = creal(M[0][1]) + creal(M[1][0]);
    components[1] = -cimag(M[0][1]) + cimag(M[1][0]);
    components[2] = M[0][0] - M[1][1];
    components[3] = creal(M[0][2]) + creal(M[2][0]);
    components[4] = -cimag(M[0][2]) + cimag(M[2][0]);
    components[5] = creal(M[1][2]) + creal(M[2][1]);
    components[6] = -cimag(M[1][2]) + cimag(M[2][1]);
    components[7] = (creal(M[0][0]) + creal(M[1][1]) -2.0*creal(M[2][2]))/sqrt(3.0);
    
    return 0;
}

struct gluon_field {
    // Structure representing the gluon_field
    int Nt;
    int Ns;
    int Nd;
    int Nc;
    double complex *A;
};

double complex pos_space_gluon_prop(struct gluon_field *gf, int mu, int nu, int a, int b, int t, int y[4]){
    // Calculate the position space gluon propagator D^ab_\mu\nu = <A^a_\mu(x)A^b_\nu(x+y)>/V
    
    int Nt = gf -> Nt;
    int Ns = gf -> Ns;
    int Nd = gf -> Nd;
    int Nc = gf -> Nc;
    
    int A_shape[DIM] = {Nt, Ns, Ns, Ns, Nd, Nc, Nc};
     
    double complex runsum = 0; // Collect running two-point sum
    
    #pragma omp parallel reduction(+: runsum) num_threads(NUM_THREADS)
    {
        #pragma omp for
        for(int i=0; i < Ns; i++){
            for(int j=0; j<Ns; j++){
                for(int k=0; k<Ns; k++){
                    double complex A_mu[Nc][Nc];
                    double complex A_nu[Nc][Nc];

                    // Initialise position at beginning of SU(N) matrix
                    int A_mu_pos[DIM] = {t,i,j,k,mu,0,0};
                    int A_nu_pos[DIM] = {t,i,j,k,nu,0,0};

                     // Translate A_\nu(x) -> A_\nu(x+y)
                    for(int yi=0; yi<Nd; yi++){
                        A_nu_pos[yi] += y[yi];
                        A_nu_pos[yi] %= A_shape[yi]; // Enforce periodicity
                    }

                    // Calculate address in memory for A position
                    int A_mu_idx = _calc_idx(A_mu_pos, A_shape, DIM);
                    int A_nu_idx = _calc_idx(A_nu_pos, A_shape, DIM);

                    // Construct gluon field matrix
                    for(int c1=0; c1<Nc; c1++){
                        for(int c2=0; c2<Nc; c2++){            
                            A_mu[c1][c2] = gf->A[A_mu_idx];
                            A_nu[c1][c2] = gf->A[A_nu_idx];

                            // Increment address
                            A_mu_idx += 1;
                            A_nu_idx += 1;
                        }
                    }

                    // Perform color decomposition
                    double complex A_mu_components[8];
                    double complex A_nu_components[8];

                    decompose_su3(A_mu,A_mu_components);
                    decompose_su3(A_nu,A_nu_components);

                    // Update running two-point sum
                    runsum += A_mu_components[a]*A_nu_components[b];
                }
            }
        }
    }
    
    return runsum/(Nt*Ns*Ns*Ns);
}

void calc_pos_space_scalar_D(struct gluon_field *gf, int t, double complex D[]){
    // Calculate the scalar propagator D in position space
    
    const int Nt = gf -> Nt;
    const int Ns = gf -> Ns;
    const int Nd = gf -> Nd;
    const int Nc = gf -> Nc;
    
    int D_shape[3] = {Ns, Ns, Ns};
    
    
    {
        for(int i=0; i<32; i++){
            for(int j=0; j<32; j++){
                for(int k=0; k<32; k++){
                    int y[4] = {t,i,j,k}; 
                    int y_pos[3] = {i,j,k}; // Position of y in D

                    int d_pos = _calc_idx(y_pos, D_shape, Nd-1);

                    // Flush D
                    D[d_pos] = 0;

                    // Loop over Lorentz indices
                    for(int mu=MU_START; mu<Nd; mu++){
                        for(int a=0; a<Nc*Nc-1; a++){    
                            D[d_pos] += gluon_prop(gf, mu, mu, a, a, t, y);
                        }
                    }
                }
            }
        }
    }
}