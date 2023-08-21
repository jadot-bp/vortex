#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define DIM 7

int _calc_idx(int pos[], int shape[], int Nd){
    // Calculate the address in memory at position pos for given shape.
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

void _calc_A(double complex U[], int pos, double complex A_mu[], int Nc){
    // Calculate the value of A_\mu corresponding to U_\mu in
    // coordinate space
    // 
    // Args:
    //
    // U[]      : Lattice data 
    // pos      : Starting address corresponding to U_\mu(x)
    // A_mu[]   : Empty container for A_\mu(x)
    // Nc       : Number of colors

    double complex trace = 0;

    for(int i=0; i < Nc; i++){
        for(int j=0; j < Nc; j++){
            A_mu[i*Nc + j] = (U[pos + i*Nc + j] - conj(U[pos + j*Nc + i]))/(2*I);
        }
        trace += A_mu[i*Nc + i];
    }

    for(int i=0; i < Nc; i++){
        A_mu[i*Nc + i] -= trace/3;
    }
}

void _decompose_SU3(double complex A[], double A_comp[], int Nc){
    // Decompose the SU3 matrix A into the Nc^2-1 Gell-Mann components
    // Only supports Nc=3 currently
    
    A_comp[0] = creal(A[1]) + creal(A[1*Nc]);
    A_comp[1] = -cimag(A[1]) + cimag(A[1*Nc]);
    A_comp[2] = A[0] - A[1*Nc + 1];
    A_comp[3] = creal(A[2]) + creal(A[2*Nc]);
    A_comp[4] = -cimag(A[2]) + cimag(A[2*Nc]);
    A_comp[5] = creal(A[1*Nc+2]) + creal(A[2*Nc+1]);
    A_comp[6] = -cimag(A[1*Nc+2]) + cimag(A[2*Nc+1]);
    A_comp[7] = (creal(A[0]) + creal(A[1*Nc+1]) -2.0*creal(A[2*Nc+2])/pow(3,0.5));
}
double complex gauge_transform(double complex U[], double complex gauge[], int Nt, int Ns, int Nd, int Nc){
    // Calculates the application of the SU(N) gauge transform over the U field.
    
    //double complex cumsum = 0;

    int U_shape[DIM] = {Nt, Ns, Ns, Ns, Nd, Nc, Nc};
    int g_shape[DIM-1] = {Nt, Ns, Ns, Ns, Nc, Nc};
    
    for(int t=0; t < Nt; t++){
        for(int i=0; i < Ns; i++){
            for(int j=0; j < Ns; j++){
                for(int k=0; k < Ns; k++){
                    for(int mu=0; mu < Nd; mu++){
                        
                        // Initialise empty temporary working matrix
                        
                        double complex tmp[Nc][Nc];
                        
                        // Calculate tmp(x) = U(x)*g(x+mu)^\dag
                        for(int c1=0; c1 < Nc; c1++){
                            for(int c2=0; c2 < Nc; c2++){                              
                                
                                tmp[c1][c2] = 0;
                                
                                for(int m=0; m < Nc; m++){ 
                                    int U_pos[DIM] = {t,i,j,k,mu,c1,m};
                                    int g_pos[DIM-1] = {t,i,j,k,c2,m};
                                    
                                    // Calculate gauge-neighbour
                                    
                                    g_pos[mu] += 1;
                                    if(mu == 0){
                                        g_pos[mu] %= Nt;
                                    }else{
                                        g_pos[mu] %= Ns;
                                    }
                                    // Accumulate product in working matrix
                                    tmp[c1][c2] += U[_calc_idx(U_pos, U_shape, DIM)]*conj(gauge[_calc_idx(g_pos, g_shape, DIM-1)]);
                                }//m loop
                            }//c2 loop
                        }//c1 loop
                        
                        // Overload U(x) with value of tmp(x) = U(x)g(x+mu)^\dag
                        
                        for(int c1=0; c1 < Nc; c1++){
                            for(int c2=0; c2 < Nc; c2++){
                                int U_pos[DIM] = {t,i,j,k,mu,c1,c2};
                               
                                U[_calc_idx(U_pos,U_shape, DIM)] = tmp[c1][c2];
                            }
                        }
                        
                        // Calculate g(x)*U'(x)
                        for(int c1=0; c1 < Nc; c1++){
                            for(int c2=0; c2 < Nc; c2++){                                                             
                                tmp[c1][c2] = 0;
                                
                                for(int m=0; m < Nc; m++){
                                    int g_pos[DIM-1] = {t,i,j,k,c1,m};
                                    int U_pos[DIM] = {t,i,j,k,mu,m,c2};
                                    
                                    // Accumulate product in working matrix
                                    tmp[c1][c2] += gauge[_calc_idx(g_pos, g_shape, DIM-1)]*U[_calc_idx(U_pos, U_shape, DIM)];
                                }//m loop
                            }//c2 loop
                        }//c1 loop
                        
                        // overload U(x) with value of tmp(x) = g(x)U(x)g(x+mu)^\dag
                        for(int c1=0; c1 < Nc; c1++){
                            for(int c2=0; c2 < Nc; c2++){
                                int U_pos[DIM] = {t,i,j,k,mu,c1,c2};
                                
                                U[_calc_idx(U_pos,U_shape, DIM)] = tmp[c1][c2];
                            }
                        }
                    }//mu loop
                }//k loop
            }//j loop
        }//i loop
    }//t loop

    return 0;
}

double evaluate_divA(double complex U[], int Nt, int Ns, int Nd, int Nc, int MU_START, double XI){
    
    double divA2 = 0;

    int U_shape[DIM] = {Nt, Ns, Ns, Ns, Nd, Nc, Nc};

    for(int t=0; t < Nt; t++){
        for(int i=0; i < Ns; i++){
            for(int j=0; j < Ns; j++){
                for(int k=0; k < Ns; k++){
    
                    double complex divA[Nc*Nc];

                    for(int c=0; c < Nc*Nc; c++){
                        divA[c] = 0;
                    }

                    for(int mu=MU_START; mu < Nd; mu++){
                        int U_pos[DIM] = {t,i,j,k,mu,0,0};
                        int U_idx = _calc_idx(U_pos, U_shape, DIM);

                        int U_neg_pos[DIM] = {t,i,j,k,mu,0,0};

                        // Step U backwards
                        if(U_pos[mu] == 0){
                            U_neg_pos[mu] = U_shape[mu]-1;
                        }else{
                            U_neg_pos[mu] -= 1;
                        }
                        int U_neg_idx = _calc_idx(U_neg_pos, U_shape, DIM);
                        
                        double complex A[Nc*Nc];
                        double complex A_neg[Nc*Nc];
                        
                        // Construct A fields
                        _calc_A(U, U_idx, A, Nc);
                        _calc_A(U, U_neg_idx, A_neg, Nc);
                        
                        // calcualte divA = sum_\mu A-A_neg
                        for(int c1=0; c1 < Nc; c1++){
                            for(int c2=0; c2 < Nc; c2++){
                                if(mu == 0){
                                    divA[c1*Nc+c2] += XI*(A[c1*Nc+c2] - A_neg[c1*Nc+c2]);
                                }else{
                                    divA[c1*Nc+c2] += A[c1*Nc+c2] - A_neg[c1*Nc+c2];
                                }
                            }
                        }
                    }

                    // Construct Gell-Mann components
                    double divA_comp[Nc*Nc-1];

                    _decompose_SU3(divA, divA_comp, Nc);

                    for(int c=0; c < Nc*Nc-1; c++){
                        divA2 += divA_comp[c]*divA_comp[c];
                    }
                }
            }
        }
    }

    return divA2/(Nt*Ns*Ns*Ns);
}

