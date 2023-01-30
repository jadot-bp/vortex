// compile via the usual gcc -o bitswap bitswap.c
// run using ./bitswap <input file> <output file> <Ns> <Nt> <1 for gauge fields or 0 for gauge_transform data>

#include <stdio.h>
#include <stdlib.h>

#define Nd 4
#define Nc 3

int main(int argc, char* argv[])
{
	FILE* fptr1;
    FILE* fptr2;
	char c;

	int i = 0;

	fptr1 = fopen(argv[1], "r");
    fptr2 = fopen(argv[2],"w");
    int Ns = atoi(argv[3]);
    int Nt = atoi(argv[4]);
    int direction = atoi(argv[5]);

	if (fptr1 == NULL) {
        printf("%s","Could not open file.");
		return 1;
	}

    int MAX_BYTES = 0;

	// Loop to read required byte
	// of file

    if (direction == 0){
        MAX_BYTES = Ns*Ns*Ns*Nt*Nc*Nc*2;
    }else{
        MAX_BYTES = Ns*Ns*Ns*Nt*Nd*Nc*Nc*2;
    }

    int j=0;
    while (j < MAX_BYTES){

        char buffer[8]; 

        int b = 0;
        for (i = j; i < j+8; i++) {

            c = fgetc(fptr1);
            buffer[b] = c;
            b++;
        }
        b = 0;

        for (int k=7; k >= 0; k--){
            fprintf(fptr2, "%c",buffer[k]);
        }
        j++;
    }

    printf("%i",j*8);

	fclose(fptr1);
    fclose(fptr2);

	return 0;
}

