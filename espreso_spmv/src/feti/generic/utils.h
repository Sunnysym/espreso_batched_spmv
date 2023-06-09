
#define PAR_VECTOR vector 
//#define PAR_VECTOR tbb::concurrent_vector

#define SEQ_VECTOR vector

//#include "mpi.h"

//#include "mkl.h"
//#include "mkl_spblas.h"
//#include "mkl_blas.h"
//#include "mkl_cblas.h"
//#include "mkl_lapacke.h"
//#include "mkl_pardiso.h"
//#include <omp.h>

//#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>

//#include <tbb/mutex.h>
//#include "tbb/parallel_sort.h"
//#include "tbb/tbb.h"

#include <string>
//#include <sstream>
//#include <iostream>
#include <vector>
//#include <fstream>
//#include <algorithm>
//#include <math.h>
//#include <iomanip>
//#include <map>
//
//#include <ctime>
//#include <stack>
//#include <time.h>
//
//#ifndef WIN32
// #include "sys/types.h"
// #include "sys/sysinfo.h"
//#endif
//
//#include <string>
//#include <sstream>
//#include <iostream>
//#include <vector>
//#include <fstream>
//#include <algorithm>
//#include <math.h>
//#include <iomanip>
//#include <map>
//
//#include "stdlib.h"
//#include "stdio.h"
//#include "string.h"

using std::vector;
//using std::map;
//using std::make_pair;
using std::string;

#pragma once

//std::stack<clock_t> tictoc_stack;


namespace espreso {

esint SaveBinVectorDouble(SEQ_VECTOR <double> & SEQ_VECTOR, string filename);
esint LoadBinVectorInt(SEQ_VECTOR <esint> & SEQ_VECTOR, string filename);
esint LoadBinVecVec(SEQ_VECTOR <SEQ_VECTOR <esint> > & outputVecVec, string filename);
esint LoadBinVecVec(SEQ_VECTOR <SEQ_VECTOR <double> > & outputVecVec, string filename);

template <typename T>
void PrintVec(SEQ_VECTOR <T> vec, string name);

template <typename T>
void PrintVecND(SEQ_VECTOR <T> vec, string name); 

void GetProcessMemoryStat_u( ); 
void GetMemoryStat_u( );  
double GetProcessMemory_u ( );

// Compare vectors and returns number of different values
int CompareVectors(SEQ_VECTOR <double>& vec_a, SEQ_VECTOR <double>& vec_b);

}
// **** END - Uncategorized functions ********************************
// *******************************************************************
