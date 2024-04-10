#!/usr/bin/env bash

OMP_FLAG=$1

STANDARD_FLAGS="-lm -Wall -pedantic -std=c99 -O3"

# Compile gprop

if [ "$OMP_FLAG" = "-omp" ]; then
    #Compile with OpenMP
    echo "Compiling with OpenMP..."
    gcc -fpic -c gprop.c $STANDARD_FLAGS -fopenmp -lgomp
    gcc -shared -lgomp -fopenmp -o libgprop.so gprop.o $STANDARD_FLAGS
else
    gcc -fpic -c gprop.c $STANDARD_FLAGS
    gcc -shared -o libgprop.so gprop.o $STANDARD_FLAGS
fi

# Compile gluon_utils

gcc -fpic -c gluon_utils.c -O3 -lm
gcc -shared -o libgutils.so gluon_utils.o -O3 -lm

# Cleanup

rm gluon_utils.o
rm gprop.o

echo "Done."
