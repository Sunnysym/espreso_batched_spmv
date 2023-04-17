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
#include "omp.h"

#include <hip/hip_runtime.h>
#include "rocsparse.h"

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

template<class dataType>
__global__ void offset(dataType* a, dataType* b,dataType* c)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    b[id] = a[id];
    c[id] = a[id];
}

template <class dataType>
struct GPUThreadArgument
{
    int gpuId;
    int StreamNum;
    vector<hipStream_t> streams;

    hipStream_t h2dStream;
    hipStream_t d2hStream;
    vector<hipEvent_t>  rs_kernel_finished;
    vector<hipEvent_t>  rs_h2d_finished;
    vector<hipEvent_t>  rs_d2h_finished;

    int** d_csrOffsets;
    int** d_columns;

    dataType** d_A;
    dataType** d_x;
    dataType** d_y;
};

int DeviceNum = 1;
int StreamNum = 16;
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
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            hipStreamCreate(&arg.streams[streamId]);
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

        int lenA = 128*1024*1024;
        int lenX = 128*1024*1024;
        int lenY = 128*1024*1024;
        int lenZ = 128*1024*1024;

        size_t ASiz = lenA * sizeof(double);
        size_t XSiz = lenX * sizeof(double);
        size_t YSiz = lenY * sizeof(double);
        size_t ZSiz = lenZ * sizeof(double);

        double *A_h, *x_h, *y_h, *z_h;
        double *A_d, *x_d, *y_d, *z_d;

        hipHostMalloc(&A_h, ASiz);
        hipHostMalloc(&x_h, XSiz);
        hipHostMalloc(&y_h, YSiz);
        hipHostMalloc(&z_h, ZSiz);

        for(int i = 0; i < lenA; i++){
            A_h[i] = i * 1.0;
            x_h[i] = i * 0.1;
        }

        hipMalloc(&A_d, ASiz);
        hipMalloc(&x_d, XSiz);
        hipMalloc(&y_d, YSiz);
        hipMalloc(&z_d, ZSiz);

        hipEvent_t h2d_A_start, h2d_A_end, h2d_start, h2d_end, kernel_start, kernel_end, d2h_start, d2h_end, d2h_Y_start, d2h_Y_end;
        hipEventCreate(&h2d_A_start);
        hipEventCreate(&h2d_A_end);
        hipEventCreate(&h2d_start);
        hipEventCreate(&h2d_end);
        hipEventCreate(&kernel_start);
        hipEventCreate(&kernel_end);
        hipEventCreate(&d2h_Y_start);
        hipEventCreate(&d2h_Y_end);
        hipEventCreate(&d2h_start);
        hipEventCreate(&d2h_end);

        int datasetsize = lenA / StreamNum;
        // 鍐峰惎鍔ㄥ鐞?
        double alltime_start = omp_get_wtime();
        //hipEventRecord(h2d_A_start, arg.streams[streamId]);
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            hipMemcpyAsync(&A_d[streamId*datasetsize], &A_h[streamId*datasetsize], ASiz/StreamNum, hipMemcpyHostToDevice, arg.streams[streamId]);
        }
        HIP_CHECK(hipDeviceSynchronize());
        double alltime_end = omp_get_wtime();
        double alldur = alltime_end - alltime_start;
        cout << alldur <<endl;
        
        alltime_start = omp_get_wtime();
        //hipEventRecord(h2d_A_start, arg.streams[streamId]);
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            hipMemcpyAsync(&x_d[streamId*datasetsize], &x_h[streamId*datasetsize], XSiz/StreamNum, hipMemcpyHostToDevice, arg.streams[streamId]);
        }
        HIP_CHECK(hipDeviceSynchronize());
        alltime_end = omp_get_wtime();
        alldur = alltime_end - alltime_start;
        cout << "x time:" << alldur <<endl;
        
        //hipEventRecord(h2d_A_end, arg.streams[streamId]);    
        /*
        hipEventRecord(h2d_start, arg.streams[streamId]);
        hipMemcpyAsync(x_d, x_h, XSiz, hipMemcpyHostToDevice, arg.streams[streamId]);
        HIP_CHECK(hipDeviceSynchronize());
        hipEventRecord(h2d_end, arg.streams[streamId]);
        */
        dim3 blocksize,gridsize;
        blocksize.x = 256;
        gridsize.x = lenX/blocksize.x;
        hipLaunchKernelGGL(offset, gridsize, blocksize, 0, 0, A_d, y_d,z_d);

        alltime_start = omp_get_wtime();
        //hipEventRecord(h2d_A_start, arg.streams[streamId]);
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            hipMemcpyAsync(&y_h[streamId*datasetsize], &y_d[streamId*datasetsize], YSiz/StreamNum, hipMemcpyDeviceToHost, arg.streams[streamId]);
        }
        HIP_CHECK(hipDeviceSynchronize());
        alltime_end = omp_get_wtime();
        alldur = alltime_end - alltime_start;
        cout << "y time:" << alldur <<endl;

        alltime_start = omp_get_wtime();
        //hipEventRecord(h2d_A_start, arg.streams[streamId]);
        for (int streamId = 0; streamId < StreamNum; ++streamId)
        {
            hipMemcpyAsync(&z_h[streamId*datasetsize], &z_d[streamId*datasetsize], ZSiz/StreamNum, hipMemcpyDeviceToHost, arg.streams[streamId]);
        }
        HIP_CHECK(hipDeviceSynchronize());
        alltime_end = omp_get_wtime();
        alldur = alltime_end - alltime_start;
        cout << "z time:" << alldur <<endl;
        /*
        hipEventRecord(kernel_start, arg.streams[streamId]);
        hipLaunchKernelGGL(offset, gridsize, blocksize, 0, 0, x_d, y_d,z_d);
        hipEventRecord(kernel_end, arg.streams[streamId]);
        HIP_CHECK(hipDeviceSynchronize());

        hipEventRecord(d2h_Y_start, arg.streams[streamId]);
        hipMemcpyAsync(y_h, y_d, YSiz, hipMemcpyDeviceToHost, arg.streams[streamId]);
        HIP_CHECK(hipDeviceSynchronize());
        hipEventRecord(d2h_Y_end, arg.streams[streamId]);

        hipEventRecord(d2h_start, arg.streams[streamId]);
        hipMemcpyAsync(z_h, z_d, ZSiz, hipMemcpyDeviceToHost, arg.streams[streamId]);
        HIP_CHECK(hipDeviceSynchronize());
        hipEventRecord(d2h_end, arg.streams[streamId]);

        float ms;
        
        hipEventElapsedTime(&ms, h2d_A_start, h2d_A_end);
        printf("device %d stream %d h2d A (ms): %.10f\n", deviceId, streamId, ms);
        hipEventElapsedTime(&ms, h2d_start, h2d_end);
        printf("device %d stream %d h2d (ms): %.10f\n", deviceId, streamId, ms);
        hipEventElapsedTime(&ms, kernel_start, kernel_end);
        printf("device %d stream %d kernel (ms): %.10f\n", deviceId, streamId, ms);
        hipEventElapsedTime(&ms, d2h_Y_start, d2h_Y_end);
        printf("device %d stream %d d2h (ms): %.10f\n", deviceId, streamId, ms);
        hipEventElapsedTime(&ms, d2h_start, d2h_end);
        printf("device %d stream %d d2h (ms): %.10f\n", deviceId, streamId, ms);
        */
            /*
            for(int i = 0; i < lenZ; i++){
                cout << z_h[i] << " ";
            }
            cout <<endl;
            */

        hipEventDestroy(h2d_A_start);
        hipEventDestroy(h2d_A_end);
        hipEventDestroy(h2d_start);
        hipEventDestroy(h2d_end);
        hipEventDestroy(kernel_start);
        hipEventDestroy(kernel_end);
        hipEventDestroy(d2h_Y_start);
        hipEventDestroy(d2h_Y_end);
        hipEventDestroy(d2h_start);
        hipEventDestroy(d2h_end);

        hipHostFree(A_h);
        hipHostFree(x_h);
        hipHostFree(y_h);
        hipHostFree(z_h);
    
        hipFree(A_d);
        hipFree(x_d);
        hipFree(y_d);
        hipFree(z_d);

    }
}


int main(int argc, char* argv[])
{
    DeviceNum = 1;
    args.resize(DeviceNum);
    gpuBoot();
    int IterMax = 10;
    for(int i = 0; i < IterMax; i++){
        gpuBoot();
    }
    return 0;
}
