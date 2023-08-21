gcc -fpic -c gluon_utils.c -O3 -lm
gcc -shared -o libgutils.so gluon_utils.o -O3 -lm

