#pragma once
#include "matrix.hpp"
#include <iostream>

#ifndef _MTX_HPP_
#define _MTX_HPP_

#define MAX_LEVELS 1000

using namespace std;

template<class dataType>
bool if_sorted_coo(MTX<dataType>* mtx)
{
	int nnz = mtx->nnz;
	for (int i = 0; i < nnz - 1; i++)
	{
		if ((mtx->row[i] > mtx->row[i + 1]) || (mtx->row[i] == mtx->row[i + 1] && mtx->col[i] >= mtx->col[i + 1]))
			return false;
	}
	return true;
}

template<class dataType>
bool sort_coo(MTX<dataType> * mtx)
{

	int i = 0;
	int beg[MAX_LEVELS], end[MAX_LEVELS], L, R;
	int pivrow, pivcol;
	dataType pivdata;

	beg[0] = 0;
	end[0] = mtx->nnz;
	while (i >= 0)
	{
		L = beg[i];
		if (end[i] - 1 > end[i])
			R = end[i];
		else
			R = end[i] - 1;
		if (L < R)
		{
			int middle = (L + R) / 2;
			pivrow = mtx->row[middle];
			pivcol = mtx->col[middle];
			pivdata = mtx->data[middle];
			mtx->row[middle] = mtx->row[L];
			mtx->col[middle] = mtx->col[L];
			mtx->data[middle] = mtx->data[L];
			mtx->row[L] = pivrow;
			mtx->col[L] = pivcol;
			mtx->data[L] = pivdata;
			if (i == MAX_LEVELS - 1)
				return false;
			while (L < R)
			{
				while (((mtx->row[R] > pivrow) ||
					(mtx->row[R] == pivrow && mtx->col[R] > pivcol))
					&& L < R)
					R--;
				if (L < R)
				{
					mtx->row[L] = mtx->row[R];
					mtx->col[L] = mtx->col[R];
					mtx->data[L] = mtx->data[R];
					L++;
				}
				while (((mtx->row[L] < pivrow) ||
					(mtx->row[L] == pivrow && mtx->col[L] < pivcol))
					&& L < R)
					L++;
				if (L < R)
				{
					mtx->row[R] = mtx->row[L];
					mtx->col[R] = mtx->col[L];
					mtx->data[R] = mtx->data[L];
					R--;
				}
			}
			mtx->row[L] = pivrow;
			mtx->col[L] = pivcol;
			mtx->data[L] = pivdata;
			beg[i + 1] = L + 1;
			end[i + 1] = end[i];
			end[i++] = L;
		}
		else
		{
			i--;
		}
	}
	return true;
}

template<class dataType>
void fileToMtx(string filename, MTX<dataType> * mtx)
{
	FILE* infile = fopen(filename.c_str(), "r");
	char tmpstr[100];
	char tmpline[1030];
	fscanf(infile, "%s", tmpstr);
	fscanf(infile, "%s", tmpstr);
	fscanf(infile, "%s", tmpstr);
	fscanf(infile, "%s", tmpstr);
	bool ifreal = false;
	if (strcmp(tmpstr, "real") == 0 || strcmp(tmpstr, "unsigned-integer") == 0 || strcmp(tmpstr, "integer") == 0)
		ifreal = true;
	bool ifsym = false;
	fscanf(infile, "%s", tmpstr);
	if (strcmp(tmpstr, "symmetric") == 0)
		ifsym = true;
	int height = 0;
	int width = 0;
	int nnz = 0;
	while (true)
	{
		fscanf(infile, "%s", tmpstr);
		if (tmpstr[0] != '%')
		{
			height = atoi(tmpstr);
			break;
		}
		fgets(tmpline, 1025, infile);
	}

	fscanf(infile, "%d %d", &width, &nnz);
	mtx->rows = height;
	mtx->cols = width;
	int* rows = (int*)malloc(sizeof(int) * nnz);
	int* cols = (int*)malloc(sizeof(int) * nnz);
	dataType* data = (dataType*)malloc(sizeof(dataType) * nnz);
	int diaCount = 0;
	for (int i = 0; i < nnz; i++)
	{
		int rowid = 0;
		int colid = 0;
		fscanf(infile, "%d %d", &rowid, &colid);
		rows[i] = rowid - 1;
		cols[i] = colid - 1;
		data[i] = i;
		if (ifreal)
		{
			double dbldata = 0.0f;
			fscanf(infile, "%lf", &dbldata);
			data[i] = (dataType)dbldata;
		}
		if (rows[i] == cols[i])
			diaCount++;
	}

	if (ifsym)
	{
		int newnnz = nnz * 2 - diaCount;
		mtx->nnz = newnnz;
		mtx->row = (int*)malloc(sizeof(int) * newnnz);
		mtx->col = (int*)malloc(sizeof(int) * newnnz);
		mtx->data = (dataType*)malloc(sizeof(dataType) * newnnz);
		int matid = 0;
		for (int i = 0; i < nnz; i++)
		{
			mtx->row[matid] = rows[i];
			mtx->col[matid] = cols[i];
			mtx->data[matid] = data[i];
			matid++;
			if (rows[i] != cols[i])
			{
				mtx->row[matid] = cols[i];
				mtx->col[matid] = rows[i];
				mtx->data[matid] = data[i];
				matid++;
			}
		}
		if (matid != newnnz) {
			cout << "Error!" << endl;
		}
		bool tmp = sort_coo<dataType>(mtx);
		if (tmp != true) {
			cout << "Error!" << endl;
		}
	}
	else
	{
		mtx->nnz = nnz;
		mtx->row = (int*)malloc(sizeof(int) * nnz);
		mtx->col = (int*)malloc(sizeof(int) * nnz);
		mtx->data = (dataType*)malloc(sizeof(dataType) * nnz);
		memcpy(mtx->row, rows, sizeof(int) * nnz);
		memcpy(mtx->col, cols, sizeof(int) * nnz);
		memcpy(mtx->data, data, sizeof(dataType) * nnz);
		if (!if_sorted_coo(mtx))
			sort_coo<dataType>(mtx);
		bool tmp = if_sorted_coo<dataType>(mtx);
		if (tmp != true) {
			cout << "Error!" << endl;
		}
	}
	fclose(infile);
	free(rows);
	free(cols);
	free(data);
	return;
}

template<class dataType>
void printMtx(MTX<dataType> * mtx)
{
	cout << "rows:" << mtx->rows << "  cols:" << mtx->cols << "  non zeros:" << mtx->nnz << endl;
	for (int i = 0; i < mtx->nnz; i++) {
		cout << mtx->col[i] << "  " << mtx->row[i] << "  " << mtx->data[i] << endl;
	}
}

template<class dataType>
int checkMtx(MTX<dataType> * mtx)
{
	for (int i = 0; i < mtx->nnz; i++) {
		if (mtx->row[i] < 0 || mtx->row[i] >= mtx->rows || mtx->col[i] < 0 || mtx->col[i] > mtx->cols) {
			cout << "Invalid index! i=" << i << ",mtx->col[i]=" << mtx->col[i] << ", mtx->row[i]=" << mtx->row[i] << ", mtx->data[i]=" << mtx->data[i] << endl;
			return 0;
		}
	}

	cout << "# check Mtx PASSED!" << endl;
	return 1;
}

#endif

template<class dataType>
void printTestMtx(MTX<dataType> * mtx)
{
	cout << "rows:" << mtx->rows << "  cols:" << mtx->cols << "  non zeros:" << mtx->nnz << endl;
	int row = mtx->row[0], row2 = 0;
	for (int i = 0; i < mtx->nnz; i++) {
		row2 = mtx->row[i];
		if (row2 != row) {
			cout << "------------------------------------------------" << endl;
			row = row2;
		}
		cout << "i=" << i << ",mtx->col[i]=" << mtx->col[i] << ", mtx->row[i]=" << mtx->row[i] << ", mtx->data[i]=" << mtx->data[i] << endl;
	}
}


template <class dataType>
void coo2csr(MTX<dataType> * mtx, CSRMatrix<dataType> * mat){
	mat->rows = mtx->rows;
    mat->cols = mtx->cols;
	mat->nnz = mtx->nnz;
	mat->columns_h = (int*) malloc(mat->nnz * sizeof(int));
	mat->csrOffsets_h = (int*) malloc((mat->rows + 1)*sizeof(int));
    mat->A_h = (dataType*) malloc(mat->nnz * sizeof(dataType));

	for (int i = 0; i < mtx->rows + 1; i++) {
		mat->csrOffsets_h[i] = 0;
	}

	for (int i = 0; i < mtx->nnz; i++)
	{
		mat->A_h[i] = mtx->data[i];

		mat->columns_h[i] = mtx->col[i];
		mat->csrOffsets_h[mtx->row[i] + 1]++;

	}
	for (int i = 0; i < mtx->rows; i++)
	{
		mat->csrOffsets_h[i + 1] += mat->csrOffsets_h[i];
	}
}

template <class dataType>
void getBatchMatrices(int batchMatNum, CSRMatrix<dataType>* matrices,int nums,vector<GPUCSRBatchMatrix<dataType>>& gpuCSRBatchMatrices){
	int batchnum = (nums - 1) / batchMatNum + 1;
	//gpuCSRBatchMatrices.resize(0);
    for (int i = 0; i < batchnum; ++i)
    {
		int colStride = 0;
		int nnzStride = 0;
		int rowStride = 0;
		//GPUCSRBatchMatrix<dataType> mat = new GPUCSRBatchMatrix<dataType>();
		//GPUCSRBatchMatrix<dataType> mat;
		int rows = 0;
		int cols = 0;
		int nnz = 0;
		vector<int> columns;
		vector<int> csrOffsets;
		vector<dataType> A;
		vector<dataType> x;
		for(int j = 0; (j < batchMatNum) && ((i * batchMatNum + j) < nums) ; j++){
			CSRMatrix<dataType> M = matrices[i * batchMatNum + j];
        	for(int k = 0; k < M.nnz; k++){
				A.push_back(M.A_h[k]);
				columns.push_back(M.columns_h[k] + colStride);
			}
			colStride += M.cols;
			for(int k = 0; k < M.rows; k++){
				csrOffsets.push_back(M.csrOffsets_h[k] + nnzStride);
			}
			nnzStride += M.nnz;
			rowStride += M.rows;
			for(int k = 0; k < M.cols; k++){
				x.push_back(M.x_h[k]);
			}
		}
		csrOffsets.push_back(nnzStride);
		gpuCSRBatchMatrices[i].M->cols = colStride;
		gpuCSRBatchMatrices[i].M->rows = rowStride;
		gpuCSRBatchMatrices[i].M->nnz = nnzStride;
		gpuCSRBatchMatrices[i].M->columns_h = (int*) malloc(sizeof(int) * nnzStride);
		gpuCSRBatchMatrices[i].M->csrOffsets_h = (int*) malloc(sizeof(int) * (rowStride + 1));
		gpuCSRBatchMatrices[i].M->A_h = (dataType*) malloc(sizeof(dataType) * nnzStride);
		gpuCSRBatchMatrices[i].M->x_h = (dataType*) malloc(sizeof(dataType) * colStride);
		if(!columns.empty()){
        	memcpy(gpuCSRBatchMatrices[i].M->columns_h,&columns[0],nnzStride*sizeof(int));
    	}
		for(int l = 0; l <nnzStride;l++){
			cout << gpuCSRBatchMatrices[i].M->columns_h[l] <<" ";
		}
		cout<<endl;
		if(!csrOffsets.empty()){
        	memcpy(gpuCSRBatchMatrices[i].M->csrOffsets_h,&csrOffsets[0],(rowStride + 1)*sizeof(int));
    	}
		if(!A.empty()){
			memcpy(gpuCSRBatchMatrices[i].M->A_h,&A[0],nnzStride*sizeof(dataType));
		}
		if(!x.empty()){
			memcpy(gpuCSRBatchMatrices[i].M->x_h,&x[0],colStride*sizeof(dataType));
		}
    }
}