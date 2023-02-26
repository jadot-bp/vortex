flag=$1

STANDARD_FLAGS="-lm -Wall -pedantic -std=c99 -O3"

if [ "$flag" = "-omp" ]; then
    #Compile with OpenMP
    echo "Compiling with OpenMp..."
    gcc -fpic -c gprop.c $STANDARD_FLAGS -fopenmp -lgomp
    gcc -shared -lgomp -fopenmp -o libgprop.so gprop.o $STANDARD_FLAGS

else
    gcc -fpic -c gprop.c $STANDARD_FLAGS
    gcc -shared -o libgprop.so gprop.o $STANDARD_FLAGS
fi

