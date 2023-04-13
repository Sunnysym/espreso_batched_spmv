#include <chrono>
#include <vector>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <map>

#include <hip/hip_runtime.h>
#include "rocsparse.h"
#include "omp.h"

#include "matrix.hpp"
#include "mtx.hpp"

using namespace std;

#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " Error: Hip reports " << hipGetErrorString(status) << std::endl; \
        std::abort(); }}

#ifndef ROCSPARSE_CHECK
#define ROCSPARSE_CHECK(status)                  \
    if(status != rocsparse_status_success)              \
    {                                                 \
        fprintf(stderr, "rocSPARSE error: ");           \
        exit(EXIT_FAILURE);                           \
    }
#endif 

template <class dataType>
struct GPUThreadArgument
{
    int gpuId;
    int StreamNum;
    vector<hipStream_t> streams;
    vector<rocsparse_handle> handles;

    vector<rocsparse_mat_descr> descrs;
    vector<rocsparse_mat_info> infos;

    hipStream_t h2dStream;
    hipStream_t d2hStream;
    vector<hipEvent_t>  rs_kernel_finished;
    vector<hipEvent_t>  rs_h2d_finished;
    vector<hipEvent_t>  rs_d2h_finished;
    
    vector<GPUCSRBatchMatrix<dataType>> mats;
    vector<GPUCSRBatchMatrix<dataType>> sharedMats;

    int** d_csrOffsets;
    int** d_columns;

    dataType** d_A;
    dataType** d_x;
    dataType** d_y;
};

int DeviceNum = 1;
int StreamNum = 4;
vector<GPUThreadArgument<double>> args;

void gpuBoot()
{
    for (int deviceId = 0; deviceId < DeviceNum; ++deviceId)
    {
        hipSetDevice(deviceId);
        GPUThreadArgument<double>& arg = args[deviceId];
        arg.gpuId = deviceId;
        arg.StreamNum = StreamNum;
        arg.streams.resize(StreamNum);
        arg.handles.resize(StreamNum);
        arg.descrs.resize(StreamNum);
        arg.infos.resize(StreamNum);
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            hipStreamCreate(&arg.streams[streamId]);
            rocsparse_create_handle(&arg.handles[streamId]);
            rocsparse_set_stream(arg.handles[streamId], arg.streams[streamId]);
            rocsparse_set_pointer_mode(arg.handles[streamId], rocsparse_pointer_mode_host);
            ROCSPARSE_CHECK(rocsparse_create_mat_descr(&arg.descrs[streamId]));
            ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(arg.descrs[streamId], rocsparse_fill_mode_lower));
            ROCSPARSE_CHECK(rocsparse_create_mat_info(&arg.infos[streamId]));
        }
        hipStreamCreate(&arg.h2dStream);
        hipStreamCreate(&arg.d2hStream);
        arg.rs_kernel_finished.resize(2);
        arg.rs_h2d_finished.resize(2);
        arg.rs_d2h_finished.resize(2);
        for (int i = 0; i < 2; ++i)
        {
            hipEventCreate(&arg.rs_kernel_finished[i]);
            hipEventCreate(&arg.rs_h2d_finished[i]);
            hipEventCreate(&arg.rs_d2h_finished[i]);
        }

        
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            string initfile = "structural_mechanics.mtx";
            MTX<double>* mtx = new MTX<double>();
            fileToMtx<double>(initfile,mtx);
            CSRMatrix<double>* mat = new CSRMatrix<double>();
            coo2csr<double>(mtx,mat);
            mat->x_h = (double*) malloc(mat->cols * sizeof(double));
            mat->y_h = (double*) malloc(mat->rows * sizeof(double));
            fill(mat->x_h, mat->x_h + mat->cols, 1);

            size_t csrSiz = (mat->rows + 1) * sizeof(int);
            size_t colSiz = mat->nnz * sizeof(int);
            size_t ASiz = mat->nnz * sizeof(double);
            size_t XSiz = mat->cols * sizeof(double);
            size_t YSiz = mat->rows * sizeof(double);

            int rows = mat->rows;
            int cols = mat->cols;
            int nnz = mat->nnz;

            int *csrOffsets_h,*columns_h;
            double *A_h, *x_h, *y_h;
            int *csrOffsets_d,*columns_d;
            double *A_d, *x_d, *y_d;

            hipHostMalloc(&csrOffsets_h, csrSiz);
            hipHostMalloc(&columns_h, colSiz);
            hipHostMalloc(&A_h, ASiz);
            hipHostMalloc(&x_h, XSiz);
            hipHostMalloc(&y_h, YSiz);

            memcpy(csrOffsets_h, mat->csrOffsets_h, sizeof(int) * (rows + 1));
            memcpy(columns_h, mat->columns_h, sizeof(int) * nnz);
            memcpy(A_h, mat->A_h, sizeof(double) * nnz);
            fill(x_h, x_h + mat->cols, 1);

            hipMalloc(&csrOffsets_d, csrSiz);
            hipMalloc(&columns_d, colSiz);
            hipMalloc(&A_d, ASiz);
            hipMalloc(&x_d, XSiz);
            hipMalloc(&y_d, YSiz);

            hipEvent_t h2d_A_start, h2d_A_end, h2d_start, h2d_end, kernel_start, kernel_end, d2h_start, d2h_end;
            hipEventCreate(&h2d_A_start);
            hipEventCreate(&h2d_A_end);
            hipEventCreate(&h2d_start);
            hipEventCreate(&h2d_end);
            hipEventCreate(&kernel_start);
            hipEventCreate(&kernel_end);
            hipEventCreate(&d2h_start);
            hipEventCreate(&d2h_end);
            
            hipEventRecord(h2d_A_start, arg.streams[streamId]);
            hipMemcpyAsync(csrOffsets_d, csrOffsets_h, csrSiz, hipMemcpyHostToDevice, arg.streams[streamId]);
            hipMemcpyAsync(columns_d, columns_h, colSiz, hipMemcpyHostToDevice, arg.streams[streamId]);
            hipMemcpyAsync(A_d, A_h, ASiz, hipMemcpyHostToDevice, arg.streams[streamId]);
            hipEventRecord(h2d_A_end, arg.streams[streamId]);
            
            hipEventRecord(h2d_start, arg.streams[streamId]);
            hipMemcpyAsync(x_d, x_h, XSiz, hipMemcpyHostToDevice, arg.streams[streamId]);
            hipEventRecord(h2d_end, arg.streams[streamId]);

            double alpha = 1.0, beta = 0;
            
            rocsparse_mat_descr descr;
            ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));
            ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));
            rocsparse_mat_info info;
            ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));
            
            
            ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(arg.handles[streamId], rocsparse_operation_none,
                rows, cols, nnz,
                arg.descrs[streamId], A_d, csrOffsets_d, columns_d, 
                arg.infos[streamId]));
             

            hipEventRecord(kernel_start, arg.streams[streamId]);
            rocsparse_dcsrmv(arg.handles[streamId], rocsparse_operation_none,
                rows, cols, nnz,
                &alpha, arg.descrs[streamId], A_d, csrOffsets_d, columns_d, 
                arg.infos[streamId], x_d,
                &beta, y_d);
            hipEventRecord(kernel_end, arg.streams[streamId]);

            hipEventRecord(d2h_start, arg.streams[streamId]);
            hipMemcpyAsync(y_h, y_d, YSiz, hipMemcpyDeviceToHost, arg.streams[streamId]);
            hipEventRecord(d2h_end, arg.streams[streamId]);

            float ms;
            /*
            hipEventElapsedTime(&ms, h2d_A_start, h2d_A_end);
            printf("device %d stream %d h2d A (ms): %.3f\n", deviceId, streamId, ms);
            hipEventElapsedTime(&ms, h2d_start, h2d_end);
            printf("device %d stream %d h2d (ms): %.3f\n", deviceId, streamId, ms);
            hipEventElapsedTime(&ms, kernel_start, kernel_end);
            printf("device %d stream %d kernel (ms): %.3f\n", deviceId, streamId, ms);
            hipEventElapsedTime(&ms, d2h_start, d2h_end);
            printf("device %d stream %d d2h (ms): %.3f\n", deviceId, streamId, ms);
            */
            ROCSPARSE_CHECK(rocsparse_csrmv_clear(arg.handles[streamId],arg.infos[streamId]));
            ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(arg.descrs[streamId]));
            ROCSPARSE_CHECK(rocsparse_destroy_mat_info(arg.infos[streamId]));

            hipEventDestroy(h2d_A_start);
            hipEventDestroy(h2d_A_end);
            hipEventDestroy(h2d_start);
            hipEventDestroy(h2d_end);
            hipEventDestroy(kernel_start);
            hipEventDestroy(kernel_end);
            hipEventDestroy(d2h_start);
            hipEventDestroy(d2h_end);

            hipHostFree(csrOffsets_h);
            hipHostFree(columns_h);
            hipHostFree(A_h);
            hipHostFree(x_h);
            hipHostFree(y_h);

            hipFree(csrOffsets_d);
            hipFree(columns_d);           
            hipFree(A_d);
            hipFree(x_d);
            hipFree(y_d);
            
            free(mat->csrOffsets_h);
            free(mat->columns_h);
            free(mat->A_h);
            free(mat->x_h);
            free(mat->y_h);
            free(mat);

            free(mtx->col);
            free(mtx->row);
            free(mtx->data);
            free(mtx);
        }
    }
}

int main(int argc, char* argv[])
{
    DeviceNum = 1;
    args.resize(DeviceNum);
    gpuBoot();

    int IterMax = 102;
    int MatNumMax = 1024;

    int nnznum = 0;
    int rownum = 0;
    int colnum = 0;

    int batchMatNum = 256;

    int Diffmtxnum = 1;
    int ifmtxs = 1;
 
    vector<string> mtxnames;

    if(ifmtxs == 1){
        string ifilename = (string)argv[1];
        ifstream in(ifilename);
        in >> MatNumMax;
        mtxnames.resize(MatNumMax);
        for (int i = 0; i < MatNumMax; ++i) in >> mtxnames[i];
        in.close();
    }
    else{
        mtxnames.resize(1);
        mtxnames[0] =  "structural_mechanics.mtx";
    }

    int* matrix_order = (int*)malloc(sizeof(int) * MatNumMax);
    //batchMatNum = (MatNumMax - 1) / StreamNum + 1;
    //cout << "batchMatNum:" << batchMatNum << endl;
    //string filename = (string)argv[1];
    //cout << "filenames,rows,cols,nnz,kerneltime,kernelperformance,alltime" << endl;
    for (int diffmtxnum = 0; diffmtxnum < Diffmtxnum; diffmtxnum += 1)
    {
        //cout << mtxnames[diffmtxnum] << endl;
        //cout << filename << endl;
        vector<CSRMatrix<double>> matrices(MatNumMax);
        map<int, vector<int>> m;
        for (int matId = 0; matId < MatNumMax; ++matId)
        {
            CSRMatrix<double>* mat = &matrices[matId];

            //string mtxnames = (string)argv[1];
            //string mtxnames = "structural_mechanics.mtx";
            MTX<double>* mtx = new MTX<double>();
            fileToMtx<double>(mtxnames[matId],mtx);

            map<int, vector<int>>::iterator iter = m.find(mtx->rows);
            if (iter != m.end()) {
                iter->second.push_back(matId);
            }
            else {
                vector<int> v(1, matId);
                m.insert(make_pair(mtx->rows, v));
            }
            
            //cout << mtxnames[matId]; 
            //if(matId == 0){
            nnznum += mtx->nnz;
            rownum += mtx->rows;
            colnum += mtx->cols;

            /*
            cout << "," << mtx->rows << ",";
            cout << mtx->cols << ",";
            cout << mtx->nnz;
            cout << endl;
                
                cout << mtx->rows << endl;
                cout << mtx->cols << endl;
                cout << mtx->nnz << endl; 
                */
            //}
            coo2csr<double>(mtx,mat);

            mat->x_h = (double*) malloc(mat->cols * sizeof(double));
            mat->y_h = (double*) malloc(mat->rows * sizeof(double));
            fill(mat->x_h, mat->x_h + mat->cols, 1);
        }

        map<int, vector<int>>::iterator it;
        int ordernum = MatNumMax - 1;
        for(it = m.begin(); it != m.end(); it++){
            vector<int>::iterator it2;
            for (it2 = (*it).second.begin(); it2 != (*it).second.end(); it2++)
            {
                matrix_order[ordernum--] = (*it2);
            }
        }

        int gpuMatNum = MatNumMax / DeviceNum;
        int batchNum = (gpuMatNum - 1) / batchMatNum + 1;
        //cout << "batchNum:" << batchNum << endl;
        for (int deviceId = 0; deviceId < DeviceNum; ++deviceId)
        {
            HIP_CHECK(hipSetDevice(deviceId));
            GPUThreadArgument<double>& arg = args[deviceId];
            arg.mats.resize(batchNum);
            int gpuMatIdOffset = deviceId * gpuMatNum;
            //getBatchMatrices<double>(batchMatNum,&matrices[gpuMatIdOffset],gpuMatNum,arg.mats);
            //cout << "after sort:" << endl;
            for (int i = 0; i < batchNum; ++i)
            {
                int colStride = 0;
                int nnzStride = 0;
                int rowStride = 0;
                //GPUCSRBatchMatrix<dataType> mat = new GPUCSRBatchMatrix<dataType>();
                //GPUCSRBatchMatrix<dataType> mat;
                GPUCSRBatchMatrix<double>& mat = arg.mats[i];
                mat.M = new CSRMatrix<double>();
                int rows = 0;
                int cols = 0;
                int nnz = 0;
                for(int j = 0; (j < batchMatNum) && ((i * batchMatNum + j) < gpuMatNum) ; j++){
                    CSRMatrix<double>* M = &matrices[matrix_order[gpuMatIdOffset + i * batchMatNum + j]];
                    cols += M->cols;
                    nnz += M->nnz;
                    rows += M->rows;
                }
                mat.M->cols = cols;
                mat.M->rows = rows;
                mat.M->nnz = nnz;
                mat.M->columns_h = (int*) malloc(sizeof(int) * nnz);
                mat.M->csrOffsets_h = (int*) malloc(sizeof(int) * (rows + 1));
                mat.M->A_h = (double*) malloc(sizeof(double) * nnz);
                mat.M->x_h = (double*) malloc(sizeof(double) * cols);
                mat.M->y_h = (double*) malloc(sizeof(double) * rows);
                rows = 0;
                cols = 0;
                nnz = 0;
                for(int j = 0; (j < batchMatNum) && ((i * batchMatNum + j) < gpuMatNum) ; j++){
                    CSRMatrix<double>* M = &matrices[matrix_order[gpuMatIdOffset + i * batchMatNum + j]];
                    //cout << "rows:" << M->rows <<" nnz:" << M->nnz << endl;
                    for(int k = 0; k < M->nnz; k++){
                        mat.M->A_h[nnz] = M->A_h[k];
                        mat.M->columns_h[nnz++] = M->columns_h[k] + colStride;
                    }
                    colStride += M->cols;
                    for(int k = 0; k < M->rows; k++){
                        mat.M->csrOffsets_h[rows++] = M->csrOffsets_h[k] + nnzStride;
                    }
                    nnzStride += M->nnz;
                    rowStride += M->rows;
                    for(int k = 0; k < M->cols; k++){
                        mat.M->x_h[cols++] = M->x_h[k];
                    }
                }
                mat.M->csrOffsets_h[rows++] = nnzStride;
            }


            for (int i = 0; i < batchNum; ++i)
            {
                GPUCSRBatchMatrix<double>& mat = arg.mats[i];
                CSRMatrix<double>* M = mat.M;

                size_t csrSiz = (M->rows + 1) * sizeof(int);
                size_t colSiz = M->nnz * sizeof(int);
                size_t ASiz = M->nnz * sizeof(double);
                size_t XSiz = M->cols * sizeof(double);
                size_t YSiz = M->rows * sizeof(double);
                
                HIP_CHECK(hipHostRegister(M->csrOffsets_h, csrSiz, hipHostRegisterPortable));
                HIP_CHECK(hipHostRegister(M->columns_h, colSiz, hipHostRegisterPortable));
                HIP_CHECK(hipHostRegister(M->A_h, ASiz, hipHostRegisterPortable));
                HIP_CHECK(hipHostRegister(M->x_h, XSiz, hipHostRegisterPortable));
                HIP_CHECK(hipHostRegister(M->y_h, YSiz, hipHostRegisterPortable));

                HIP_CHECK(hipMalloc(&mat.csrOffsets_d, csrSiz));
                HIP_CHECK(hipMalloc(&mat.columns_d, colSiz));                
                HIP_CHECK(hipMalloc(&mat.A_d, ASiz));
                HIP_CHECK(hipMalloc(&mat.x_d, XSiz));
                HIP_CHECK(hipMalloc(&mat.y_d, YSiz));

                HIP_CHECK(hipMemcpyAsync(mat.csrOffsets_d, M->csrOffsets_h, csrSiz, hipMemcpyHostToDevice, arg.h2dStream));
                HIP_CHECK(hipMemcpyAsync(mat.columns_d, M->columns_h, colSiz, hipMemcpyHostToDevice, arg.h2dStream));               
                HIP_CHECK(hipMemcpyAsync(mat.A_d, M->A_h, ASiz, hipMemcpyHostToDevice, arg.h2dStream));
                
            }

            hipMalloc(&arg.d_csrOffsets, sizeof(int*) * batchNum);
            hipMalloc(&arg.d_columns, sizeof(int*) * batchNum);
            hipMalloc(&arg.d_A, sizeof(double*) * batchNum);
            hipMalloc(&arg.d_x, sizeof(double*) * batchNum);
            hipMalloc(&arg.d_y, sizeof(double*) * batchNum);

            for (int matId = 0; matId < batchNum; ++matId)
            {
                GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                arg.d_csrOffsets[matId] = mat.csrOffsets_d;
                arg.d_columns[matId] = mat.columns_d; 
                arg.d_A[matId] = mat.A_d;
                arg.d_x[matId] = mat.x_d;
                arg.d_y[matId] = mat.y_d;
            } 

            HIP_CHECK(hipDeviceSynchronize());
            
            arg.descrs.resize(batchNum);
            arg.infos.resize(batchNum);
            for (int matId = 0; matId < batchNum; ++matId)
            {
                ROCSPARSE_CHECK(rocsparse_create_mat_descr(&arg.descrs[matId]));
                ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(arg.descrs[matId], rocsparse_fill_mode_lower));
                ROCSPARSE_CHECK(rocsparse_create_mat_info(&arg.infos[matId]));
                int streamId = matId % StreamNum;
                GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                CSRMatrix<double>* M = mat.M;
                    
            }
            HIP_CHECK(hipDeviceSynchronize());
        }

        vector<double> times(IterMax);
        vector<double> atimes(IterMax);
        for (int iter = 0; iter < IterMax; ++iter)
        {
            vector<double> alltimes(DeviceNum);
            vector<double> kertimes(DeviceNum);

            #pragma omp parallel for
            for (int deviceId = 0; deviceId < DeviceNum; ++deviceId)
            {
                HIP_CHECK(hipSetDevice(deviceId));
                GPUThreadArgument<double>& arg = args[deviceId];

                double alpha = 1.0;
                double beta  = 0.0;
                if(iter < 1){
                    for (int matId = 0; matId < batchNum; ++matId)
                    {
                        int streamId = matId % StreamNum;
                        GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                        CSRMatrix<double>* M = mat.M;
                            
                        ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(arg.handles[streamId], rocsparse_operation_none,
                        M->rows, M->cols, M->nnz,
                        arg.descrs[matId], mat.A_d, mat.csrOffsets_d, mat.columns_d, 
                        arg.infos[matId]));
                            
                    }
                }
                HIP_CHECK(hipDeviceSynchronize());
                
                double alltime_start = omp_get_wtime();
                for (int matId = 0; matId < batchNum; ++matId)
                {
                    GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                    CSRMatrix<double>* M = mat.M;
                    size_t XSiz = M->cols * sizeof(double);
                    HIP_CHECK(hipMemcpyAsync(mat.x_d, M->x_h, XSiz, hipMemcpyHostToDevice, arg.h2dStream));
                }
                HIP_CHECK(hipDeviceSynchronize());

                double kertime_start = omp_get_wtime();
                for (int matId = 0; matId < batchNum; ++matId)
                {
                    int streamId = matId % StreamNum;
                    GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                    CSRMatrix<double>* M = mat.M;

                    rocsparse_dcsrmv(arg.handles[streamId], rocsparse_operation_none,
                                        M->rows, M->cols, M->nnz,
                                        &alpha, arg.descrs[matId], mat.A_d, mat.csrOffsets_d, mat.columns_d, 
                                        arg.infos[matId], mat.x_d,
                                        &beta, mat.y_d);
                }

                HIP_CHECK(hipDeviceSynchronize());
                double kertime_end = omp_get_wtime();
                double kerdur = kertime_end - kertime_start;
                kertimes[deviceId] = kerdur;

                for (int matId = 0; matId < batchNum; ++matId)
                {
                    int streamId = matId % StreamNum;
                    GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                    CSRMatrix<double>* M = mat.M;
                    size_t YSiz = M->rows * sizeof(double);
                    HIP_CHECK(hipMemcpyAsync(M->y_h, mat.y_d, YSiz, hipMemcpyDeviceToHost, arg.d2hStream));
                }
                hipDeviceSynchronize();
                double alltime_end = omp_get_wtime();
                double alldur = alltime_end - alltime_start;
                alltimes[deviceId] = alldur;
                /*
                for (int matId = 0; matId < batchNum; ++matId)
                {
                    int streamId = matId % StreamNum;
                    GPUCSRBatchMatrix<double>& mat = arg.mats[matId];
                    CSRMatrix<double>* M = mat.M;
                    cout<<endl;
                    for(int j=0;j<M->rows;j++){
                        cout<<M->y_h[j]<<" ";
                    }
                    cout<<endl;
                }
                */
            }

            double tsum = std::accumulate(kertimes.begin(), kertimes.end(), 0.);
            times[iter] = tsum / DeviceNum;
            tsum = std::accumulate(alltimes.begin(), alltimes.end(), 0.);
            atimes[iter] = tsum / DeviceNum;
        }

        double time_sum = std::accumulate(times.begin() + 2, times.end(), 0.);
        double time_avg = time_sum / (IterMax - 2);
        //performance for different matrix
        double performance = nnznum * 2 * 1.0; 
        performance = performance / (time_avg * 1e9);
        //performance for the same matrix
        //double performance = (MatNumMax) * 2 * nnznum * 1.;
        //performance = performance / (time_avg * 1e9);
        //printf(",%.10f,%.6f,", time_avg, performance);
        printf("%.10f\n%.6f\n", time_avg, performance);
        time_sum = std::accumulate(atimes.begin() + 2,  atimes.end(), 0.);
        time_avg = time_sum / (IterMax - 2);
        printf("%.10f\n", time_avg);

        for (int deviceId = 0; deviceId < DeviceNum; ++deviceId)
        {
            hipSetDevice(deviceId);
            GPUThreadArgument<double>& arg = args[deviceId];

            for (int matId = 0; matId < batchNum; ++matId)
            {
                int streamId = matId % StreamNum;
                ROCSPARSE_CHECK(rocsparse_csrmv_clear(arg.handles[streamId],arg.infos[matId]));
                ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(arg.descrs[matId]));
                ROCSPARSE_CHECK(rocsparse_destroy_mat_info(arg.infos[matId]));
            }

            for (int i = 0; i < batchNum; ++i)
            {
                GPUCSRBatchMatrix<double>& mat = arg.mats[i];
                CSRMatrix<double>* M = mat.M;

                HIP_CHECK(hipHostUnregister(M->csrOffsets_h));
                HIP_CHECK(hipHostUnregister(M->columns_h));
                HIP_CHECK(hipHostUnregister(M->A_h));
                HIP_CHECK(hipHostUnregister(M->x_h));
                HIP_CHECK(hipHostUnregister(M->y_h));
                
                HIP_CHECK(hipFree(mat.csrOffsets_d));
                HIP_CHECK(hipFree(mat.columns_d));
                HIP_CHECK(hipFree(mat.A_d));
                HIP_CHECK(hipFree(mat.x_d));
                HIP_CHECK(hipFree(mat.y_d));
            }
            HIP_CHECK(hipFree(arg.d_csrOffsets));
            HIP_CHECK(hipFree(arg.d_columns));
            HIP_CHECK(hipFree(arg.d_A));
            HIP_CHECK(hipFree(arg.d_y));
            HIP_CHECK(hipFree(arg.d_x));
        }

        for (int matId = 0; matId < MatNumMax; ++matId)
        {
            CSRMatrix<double>& mat = matrices[matId];
            free(mat.csrOffsets_h);
            free(mat.columns_h);
            free(mat.A_h);
            free(mat.x_h);
            free(mat.y_h);

        }
    }
    free(matrix_order);
    return 0;
}
