#include <stdlib.h>
#include <complex.h>

#ifndef __gprop_h__
#define __gprop_h__

double complex gluon_prop(struct gluon_field gf, int mu, int nu, int a, int b, int t, int y[]);
void calc_scalar_D(struct gluon_field gf, int t, double complex D[]);
void pragma_test();

#endif


