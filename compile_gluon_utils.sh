gcc -fpic -c gluon_utils.c
gcc -shared -o libgutils.so gluon_utils.o

