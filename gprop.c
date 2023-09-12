#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>

#define DIM 7

#define PI 3.14159265358979323846

int _calc_idx(int pos[], int shape[], int dim){
    // Calculate the row-major address in  memory at position pos for given shape.
    //
    // Args:
    //
    // pos[]   : Position within the array
    // shape[] : Shape (dimensionality) of the array
    // Nd      : Number of dimensions of the array.
        
    int idx = 0;
    
    for(int k=0; k < dim; k++){
        int prod_dim = 1;
        
        for(int l=k+1; l < dim; l++){
            prod_dim *= shape[l];
        }
        idx += prod_dim*pos[k];
    }
    return idx;
}

void decompose_su3(double complex M[3][3], double complex components[8]){
    // Decompose the matrix M[][] into the SU(3) Gell-Mann components, returned in components[]
    
    components[0] = creal(M[0][1]) + creal(M[1][0]);
    components[1] = -cimag(M[0][1]) + cimag(M[1][0]);
    components[2] = M[0][0] - M[1][1];
    components[3] = creal(M[0][2]) + creal(M[2][0]);
    components[4] = -cimag(M[0][2]) + cimag(M[2][0]);
    components[5] = creal(M[1][2]) + creal(M[2][1]);
    components[6] = -cimag(M[1][2]) + cimag(M[2][1]);
    components[7] = (creal(M[0][0]) + creal(M[1][1]) -2.0*creal(M[2][2]))/sqrt(3.0);
}

void trless_conj_subtract(double complex A[3][3], double complex B[3][3], double complex M[3][3]){
    // Calculate the traceless conjugate subtraction of A[][] and B[][], returned in M[][]
    //
    // M = (A-B) - Tr(A-B)/Dim(A)
    
    double complex trace = 0;
    
    for(int i=0; i<3; i++){
        trace += A[i][i] - conj(B[i][i]);
    }
    
    trace /= 3;

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            M[i][j] = A[i][j] - conj(B[j][i]);
        }
    }
    for(int i=0; i<3; i++){
        M[i][i] -= trace;
    }
}

void scalar_matmul(double complex scalar, double complex A[3][3], double complex M[3][3]){
    // Multiply a matrix A[][] by a scalar, returned in M[][]

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            M[i][j] = scalar*A[i][j];
        }
    }
}

struct gluon_field {
    // Structure representing the gluon_field
    int Nt;
    int Ns;
    int Nd;
    int Nc;
    double complex *U;
};

void calc_mom_space_scalarD(struct gluon_field *gf, double complex D[], double complex D4[], int MU_START){
    // Calculate the scalar propagator D in position space
    
    const int Nt = gf -> Nt;
    const int Ns = gf -> Ns;
    const int Nd = gf -> Nd;
    const int Nc = gf -> Nc;

    int U_shape[DIM] = {Nt, Ns, Ns, Ns, Nd, Nc, Nc};
 
    int counter = 0;

    for(int t=0; t<Nt/2; t++){
        for(int qx=0; qx<Ns/2; qx++){
            for(int qy=0; qy<Ns/2; qy++){
                for(int qz=0; qz<Ns/2; qz++){
                    
                    D[counter] = 0.0;
                    D4[counter] = 0.0;

                    for(int mu=MU_START; mu<Nd; mu++){
                        double complex U_pos[Nc][Nc];
                        double complex U_neg[Nc][Nc];
                        
                        // Initialise position at beginning of SU(N) matrix
                        int U_pos_coord[DIM] = {t,qx,qy,qz,mu,0,0};
                        int U_neg_coord[DIM] = {(Nt-t)%Nt,(Ns-qx)%Ns,(Ns-qy)%Ns,(Ns-qz)%Ns,mu,0,0};

                        // Calculate address in memory for A position
                        int U_pos_idx = _calc_idx(U_pos_coord, U_shape, DIM);
                        int U_neg_idx = _calc_idx(U_neg_coord, U_shape, DIM);
                        
                        // Construct link field matrix
                        
                        for(int c1=0; c1<Nc; c1++){
                            for(int c2=0; c2<Nc; c2++){            
                                U_pos[c1][c2] = gf->U[U_pos_idx];
                                U_neg[c1][c2] = gf->U[U_neg_idx];

                                // Increment address
                                U_pos_idx += 1;
                                U_neg_idx += 1;
                            }
                        }
                        
                        // Calculate gluon field matrix
                        
                        double complex A[Nc][Nc];
                        double complex A_neg[Nc][Nc];
                        
                        trless_conj_subtract(U_pos,U_neg,A);
                        trless_conj_subtract(U_neg,U_pos,A_neg);
                         
                        scalar_matmul(cexp(-PI*I*U_pos_coord[mu]/U_shape[mu])/(2*I), A, A);
                        scalar_matmul(cexp(PI*I*U_pos_coord[mu]/U_shape[mu])/(2*I), A_neg, A_neg);

                        // Perform color decomposition
                        double complex A_components[8];
                        double complex A_neg_components[8];

                        decompose_su3(A,A_components);
                        decompose_su3(A_neg,A_neg_components);

                        // Update running two-point sum
                        for(int i=0; i<8; i++){
                            D[counter] += A_components[i]*A_neg_components[i];
                        }
                        // Update D4 if mu = 0 for anisotropic Landau
                        if(mu == 0){
                            for(int i=0; i<8; i++){
                                D4[counter] += A_components[i]*A_neg_components[i];
                            }
                        }
                           
                        
                    }// mu loop
                    
                    // Multiply by prefactors:
                    
                    double prefactor;

                    if(MU_START == 1 && qx+qy+qz < 1){
                        prefactor = 2.0/((Nc*Nc-1)*Nd*Nt*Ns*Ns*Ns);
                    }else if(MU_START == 0 && t+qx+qy+qz < 1){
                        prefactor = 2.0/((Nc*Nc-1)*Nd*Nt*Ns*Ns*Ns);
                    }else{
                        prefactor = 2.0/((Nc*Nc-1)*(Nd-1)*Nt*Ns*Ns*Ns);
                    }                    
                    D[counter] *= prefactor;
                    if(MU_START == 0){
                        D4[counter] *= prefactor;
                    }
                    
                    counter += 1;
                }//qz loop
            }//qy loop    
        }//qx loop
    }//t loop
    
}//program
