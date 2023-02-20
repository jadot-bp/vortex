#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

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
