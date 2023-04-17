#include <stdio.h>
#include <iostream> 
#include <stdlib.h>
#include <cstring>
#include <float.h>

#include "mtx.hpp"
#include "class.hpp"
#include "getfeatures.hpp"

using namespace std;

int main(int argc, char* argv[])
{
	char* filename = (char*)argv[1];
	//char filename[100] = "srtuctural_mechanics.mtx";
	FILE* infile = fopen(filename, "r");
	cout << filename << endl;

	//fileToMtx<float>(filename, &mtx);
	//printMtx<float>(&mtx);

	MTX<double> mtx;

	int file_test = 1;
	if (file_test == 1) {
		fileToMtx<double>(filename, &mtx);
	}
	else {
		mtx.rows = 100;
		mtx.cols = 100;
		mtx.nnz = 28;
		mtx.row = new int[mtx.nnz];
		mtx.col = new int[mtx.nnz];
		mtx.data = new double[mtx.nnz];
		for (int i = 0; i < 3; i++) {
			mtx.row[i * 2] = i;
			mtx.row[i * 2 + 1] = i;
		}
		mtx.col[0] = 2;
		mtx.col[1] = 3;
		mtx.col[2] = 2;
		mtx.col[3] = 3;
		mtx.col[4] = 4;
		mtx.col[5] = 5;

		mtx.row[6] = 2;
		mtx.col[6] = 6;
		mtx.row[7] = 2;
		mtx.col[7] = 7;
		mtx.row[8] = 2;
		mtx.col[8] = 8;

		for (int i = 9; i <= 12; i++) {
			mtx.row[i] = 6;
		}
		mtx.col[9] = 2;
		mtx.col[10] = 3;
		mtx.col[11] = 10;
		mtx.col[12] = 11;

		mtx.row[13] = 8;
		mtx.col[13] = 12;
		mtx.row[14] = 8;
		mtx.col[14] = 13;
		mtx.row[15] = 8;
		mtx.col[15] = 14;
		mtx.row[16] = 8;
		mtx.col[16] = 15;
		mtx.row[17] = 11;
		mtx.col[17] = 1;

		for (int i = 18; i <= 27; i++) {
			mtx.row[i] = 19;
			mtx.col[i] = i;
		}
		mtx.col[25] = 40;
		mtx.col[26] = 41;
		mtx.col[27] = 80;

		for (int i = 0; i < mtx.nnz; i++) {
			mtx.data[i] = (mtx.col[i] + mtx.row[i]) / 100.0;
		}

	}

	int block_size = 8;
	//printMtx(&mtx);
	printFeatures(&mtx, block_size);
	return 0;
}