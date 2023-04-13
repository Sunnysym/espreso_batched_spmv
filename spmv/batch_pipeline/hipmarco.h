#pragma once
#include <iostream>
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
