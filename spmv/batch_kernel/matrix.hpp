#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>

#include "hip/hip_runtime.h"

#include "hipmarco.h"

using namespace std;

template <class dataType>
struct MTX {
	int rows, cols, nnz;
	int *row, *col;
	dataType *data;
};

template <class dataType>
struct CSRMatrix
{
    int id;
    int rows, cols, nnz;
    int *columns_h, *csrOffsets_h;
    dataType *A_h, *x_h, *y_h;
};

template <class dataType>
struct GPUCSRMatrix
{
    CSRMatrix<dataType>* M;
    int *columns_d, *csrOffsets_d;
    dataType *A_d, *x_d, *y_d;
};

template <class dataType>
struct GPUCSRBatchMatrix
{
    CSRMatrix<dataType>* M;
    int *columns_d, *csrOffsets_d;
    dataType *A_d, *x_d, *y_d;
};

template <class dataType>
vector<CSRMatrix<dataType>*> generateMatrices(vector<string>& filenames);
template <class dataType>
vector<GPUCSRMatrix<dataType>*> generateMatricesOnGPU(vector<CSRMatrix<dataType>*>& matrices);
template <class dataType>
void freeCSRMatricesOnGPU(vector<GPUCSRMatrix<dataType>*>& matrices);
template <class dataType>
void freeCSRMatrices(vector<CSRMatrix<dataType>*>& matrices);

template <class dataType>
vector<CSRMatrix<dataType>*> generateMatrices(vector<string>& filenames)
{
    int nums;
    nums = filenames.size();
    vector<CSRMatrix<dataType>*> matrices;
    for (int i = 0; i < nums; ++i)
    {
        MTX<dataType>* mtx = new MTX<dataType>();
        fileToMtx<dataType>(filenames[i],mtx);
        CSRMatrix<dataType>* mat = new CSRMatrix<dataType>();
        coo2csr<dataType>(mtx,mat);

        mat->x_h = (dataType*) malloc(mat->cols * sizeof(dataType));
        mat->y_h = (dataType*) malloc(mat->rows * sizeof(dataType));
        fill(mat->x_h, mat->x_h + mat->cols, 1);

        matrices.push_back(mat);

        free(mtx->col);
        free(mtx->row);
        free(mtx->data);
        free(mtx);
    }
    return matrices;
}

template <class dataType>
vector<GPUCSRMatrix<dataType>*> generateMatricesOnGPU(vector<CSRMatrix<dataType>*>& matrices)
{
    int nums = matrices.size();
    vector<GPUCSRMatrix<dataType>*> gpuCSRMatrices;
    for (int i = 0; i < nums; ++i)
    {
        CSRMatrix<dataType>* M = matrices[i];
        GPUCSRMatrix<dataType>* mat = new GPUCSRMatrix<dataType>();
        mat->M = M;
        size_t csrSiz = (M->rows + 1) * sizeof(int);
        size_t colSiz = M->nnz * sizeof(int);
        size_t ASiz = M->nnz * sizeof(dataType);
        size_t XSiz = M->cols * sizeof(dataType);
        size_t YSiz = M->rows * sizeof(dataType);
        HIP_CHECK(hipHostRegister(M->csrOffsets_h, csrSiz, hipHostRegisterPortable));
        HIP_CHECK(hipHostRegister(M->columns_h, colSiz, hipHostRegisterPortable));
        HIP_CHECK(hipHostRegister(M->A_h, ASiz, hipHostRegisterPortable));
        HIP_CHECK(hipHostRegister(M->x_h, XSiz, hipHostRegisterPortable));
        HIP_CHECK(hipHostRegister(M->y_h, YSiz, hipHostRegisterPortable));
        HIP_CHECK(hipMalloc(&mat->csrOffsets_d, csrSiz));
        HIP_CHECK(hipMalloc(&mat->columns_d, colSiz));        
        HIP_CHECK(hipMalloc(&mat->A_d, ASiz));
        HIP_CHECK(hipMalloc(&mat->x_d, XSiz));
        HIP_CHECK(hipMalloc(&mat->y_d, YSiz));
        HIP_CHECK(hipMemcpy(mat->csrOffsets_d, M->csrOffsets_h, csrSiz, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(mat->columns_d, M->columns_h, colSiz, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(mat->A_d, M->A_h, ASiz, hipMemcpyHostToDevice));
        gpuCSRMatrices.push_back(mat);
    }
    return gpuCSRMatrices;
}


template <class dataType>
void freeCSRMatricesOnGPU(vector<GPUCSRMatrix<dataType>*>& matrices)
{
    int nums = matrices.size();
    for (int i = 0; i < nums; ++i)
    {
        GPUCSRMatrix<dataType>* mat = matrices[i];
        HIP_CHECK(hipFree(mat->csrOffsets_d));
        HIP_CHECK(hipFree(mat->columns_d));
        HIP_CHECK(hipFree(mat->A_d));
        HIP_CHECK(hipFree(mat->x_d));
        HIP_CHECK(hipFree(mat->y_d));
        free(mat);
    }
}

template <class dataType>
void freeCSRMatrices(vector<CSRMatrix<dataType>*>& matrices)
{
    int nums = matrices.size();
    for (int i = 0; i < nums; ++i)
    {
        CSRMatrix<dataType>* mat = matrices[i];
        HIP_CHECK(hipHostUnregister(mat->csrOffsets_h));
        HIP_CHECK(hipHostUnregister(mat->columns_h));
        HIP_CHECK(hipHostUnregister(mat->A_h));
        HIP_CHECK(hipHostUnregister(mat->x_h));
        HIP_CHECK(hipHostUnregister(mat->y_h));
        free(mat->csrOffsets_h);
        free(mat->columns_h);
        free(mat->A_h);
        free(mat->x_h);
        free(mat->y_h);
        free(mat);
    }
}
