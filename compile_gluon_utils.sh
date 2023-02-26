gcc -fpic -c gluon_utils.c -O3
gcc -shared -o libgutils.so gluon_utils.o -O3

