#pragma once
#include "class.hpp"
#include "mtx.hpp"
#include <limits.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#ifndef _GETFEATURES_HPP_
#define _GETFEATURES_HPP_


using namespace std;

double average(int *x, int len){
    int sum = 0;
    for(int i = 0; i < len; i++)
    {
        sum +=x[i];
    }
    double ave = sum * 1.0 / len; 
    return ave;
}

double getvariance(int *x, int ave, int len){
    double sum = 0;
    for(int i = 0; i < len; i++){
        sum += pow(x[i] - ave, 2);
    }
    return sum * 1.0 / len;
}

double getstandardDev(int *x, int ave, int len){
    double variance = getvariance(x, ave, len);
    return sqrt(variance);
}

int getmin(int *x, int len){
    int min = INT_MAX;
    for(int i = 0; i < len; i++){
        if(x[i] < min)
            min = x[i];
    }
    return min;
}

int getmax(int *x, int len){
    int max = INT_MIN;
    for(int i = 0; i < len; i++){
        if(x[i] > max)
            max = x[i];
    }
    return max;
}

int getne(int *x, int len){
    int sum = 0;
    for(int i = 0; i < len; i++){
        if(x[i] != 0)
            sum++;
    }
    return sum;
}

int getone(int *x, int len){
    int sum = 0;
    for(int i = 0; i < len; i++){
        if(x[i] == 1)
            sum++;
    }
    return sum;
}

double getgini(int *x, int len){
    double numerator = 0;
    double denominator = 0;
    for(int i = 0; i < len; i++){
        numerator += x[i]*2*len;
        for(int j = 0; j < len; j++){
            denominator += abs(x[i] - x[j]);
        }
    }
    double result = denominator / numerator * 1.0;
    return result;
}

double getpratio(int *x, int len, int nnznum){
    vector<int> arr(x,x+len);
    sort(arr.begin(),arr.end());
    double metaratio = 1.0 / len;
    double pratio;
    int sum = 0;
    for(int i = len - 1; i >= 0 ; i--){
        sum += arr[i];
        pratio = metaratio * (len - i);
        double nnzratio = sum * 1.0 / nnznum;
        if(pratio + nnzratio > 1)
        {
            double last_pratio = metaratio * (len - i - 1);
            double last_nnzratio = (sum-arr[i]) * 1.0 / nnznum;
            double gap =  pratio + nnzratio - 1;
            double last_gap = last_nnzratio + last_pratio - 1;
            if(gap > last_gap)
            {
                pratio = last_pratio;
                nnzratio = last_nnzratio;
            }
            break;
        }
    } 
    return pratio;
}

double getdvalue_ave(int *x, int len){
    double sum = 0;
    for(int i = 1; i < len; i++){
        sum += abs(x[i] - x[i-1]);
    }
    return sum * 1.0 / (len-1);
}

template<class dataType>
double getdis(MTX<dataType> * mtx){
    double dis = 0;
    double dis_i = 0;
    int num = 0;
    int rownum = mtx->row[0];
    for(int i = 1; i < mtx->nnz; i++){
        // current nonzero is in the same line with last nonzero
        if(mtx->row[i] == rownum){
            dis_i += (mtx->col[i] - mtx->col[i-1]);
            num = num + 1;
            //cout << "\ndis:" << dis <<endl;
            if (i == (mtx->nnz - 1)){
                dis_i = dis_i / num;
                dis = dis + dis_i;
            } 
        }
        else if (num > 0){
            dis_i = dis_i / num;
            dis = dis + dis_i;
            dis_i = 0;
            num = 0;
            rownum = mtx->row[i];
        }
        else{
            rownum = mtx->row[i];
        }
    }
    dis = dis * 1.0 / mtx->rows;
    return dis;
}

template<class dataType>
void printFeatures(MTX<dataType> * mtx , int K)
{
    /*
    // get the nonzero num in average row and col
    int* row_data = (int*)malloc(sizeof(int) * mtx->rows);
	int* col_data = (int*)malloc(sizeof(int) * mtx->cols);
    memset(row_data, 0, sizeof(int) * mtx->rows);
    memset(col_data, 0, sizeof(int) * mtx->cols);
    for(int i = 0; i < mtx->nnz; i++){
        row_data[mtx->row[i]]++;
        col_data[mtx->col[i]]++;
    }

    //divide the matrix into K * K blocks, and get the nonzero num
    int rowstride = 64 / (mtx->nnz * 1.0 / mtx->rows);
    int colstride = 16;
    int coltile_num = (mtx->cols - 1) / colstride + 1;
    int rowtiles = mtx->rows * coltile_num;
    int* rowtile_data = (int*)malloc(sizeof(int) * rowtiles);

    int rowtile_num = (mtx->rows - 1) / rowstride + 1;
    int alltiles = rowtile_num  * coltile_num;
    cout<<"\nrowbin_num"<<rowbin_num<<" colbin_num"<<colbin_num<<" allbin_num"<<allbin_num<<endl;
    int* alltile_data = (int*)malloc(sizeof(int) * alltiles);
    memset(rowtile_data, 0, sizeof(int) * rowtiles);
    memset(alltile_data, 0, sizeof(int) * alltiles);
    for(int i = 0; i < mtx->nnz; i++){
        int coltileid = mtx->col[i] / colstride;
        int rowtileid = coltileid + coltile_num * mtx->row[i];
        rowtile_data[rowtileid]++;
        rowtileid = mtx->row[i] / rowstride;
        int alltileid = coltileid + coltile_num * rowtileid;
        alltile_data[alltileid]++;
    }
    double uniqr = 1 - getone(rowtile_data,rowtiles) * 1.0 / getne(rowtile_data,rowtiles);
    uniqr = uniqr * 1.0 / mtx->nnz;
    double uniqtile = 1 - getone(alltile_data,alltiles) * 1.0 /getne(alltile_data,alltiles);
    uniqtile = uniqtile * 1.0 / mtx->nnz;

    
    double row_dvalue_ave = getdvalue_ave(row_data,mtx->rows);
    double dis = getdis(mtx);
	cout << "rows:\n" << mtx->rows << "\ncols:\n" << mtx->cols << "\nnon zeros:\n" << mtx->nnz << endl;
    
    cout << "row_dvalue_ave:\n" << row_dvalue_ave << endl;
    cout << "dis:\n" << dis <<endl;
    cout << "uniqr:\n" << uniqr << endl;
    cout << "uniqtile:\n" << uniqtile << endl;

    
    // get features
    double ave_row = mtx->nnz * 1.0 / mtx->rows;
    double ave_col = mtx->nnz * 1.0 / mtx->cols;
    double standardDev_row = getstandardDev(row_data,ave_row,mtx->rows);
    double standardDev_col = getstandardDev(col_data,ave_col,mtx->cols);
    double variance_row = getvariance(row_data,ave_row,mtx->rows);
    double variance_col = getvariance(col_data,ave_col,mtx->cols);
    int min_row = getmin(row_data,mtx->rows);
    int min_col = getmin(col_data,mtx->cols);
    int max_row = getmax(row_data,mtx->rows);
    int max_col = getmax(col_data,mtx->cols);
    int ne_row = getne(row_data,mtx->rows);
    int ne_col = getne(col_data,mtx->cols);
    double gini_row = getgini(row_data,mtx->rows);
    double gini_col = getgini(col_data,mtx->cols);
    double pratio_row = getpratio(row_data,mtx->rows,mtx->nnz);
    double pratio_col = getpratio(col_data,mtx->cols,mtx->nnz);
    
    
    divide the matrix into K * K blocks, and get the nonzero num
    int rowbin_num = (mtx->rows - 1) / K + 1;
    int colbin_num = (mtx->cols - 1) / K + 1;
    int allbin_num = rowbin_num * colbin_num;
    cout<<"\nrowbin_num"<<rowbin_num<<" colbin_num"<<colbin_num<<" allbin_num"<<allbin_num<<endl;
    int* rowbin_data = (int*)malloc(sizeof(int) * rowbin_num);
	int* colbin_data = (int*)malloc(sizeof(int) * colbin_num);
    int* allbin_data = (int*)malloc(sizeof(int) * allbin_num);
    memset(rowbin_data, 0, sizeof(int) * rowbin_num);
    memset(colbin_data, 0, sizeof(int) * colbin_num);
    memset(allbin_data, 0, sizeof(int) * allbin_num);
    for(int i = 0; i < mtx->nnz; i++){
        int rowbin_id = mtx->row[i] / K;
        int colbin_id = mtx->col[i] / K;
        int allbin_id = rowbin_id * K + colbin_id;
        rowbin_data[rowbin_id]++;
        colbin_data[colbin_id]++;
        allbin_data[allbin_id]++;
    }
    
    // get features
    double ave_rowbin = mtx->nnz * 1.0 / rowbin_num;
    double ave_colbin = mtx->nnz * 1.0 / colbin_num;
    double ave_allbin = mtx->nnz * 1.0 / allbin_num;
    
    double standardDev_rowbin = getstandardDev(rowbin_data,ave_rowbin,rowbin_num);
    double standardDev_colbin = getstandardDev(colbin_data,ave_colbin,colbin_num);
    double standardDev_allbin = getstandardDev(allbin_data,ave_allbin,allbin_num);

    double variance_rowbin = getvariance(rowbin_data,ave_rowbin,rowbin_num);
    double variance_colbin = getvariance(colbin_data,ave_colbin,colbin_num);
    double variance_allbin = getvariance(allbin_data,ave_allbin,allbin_num);

    int min_rowbin = getmin(rowbin_data,rowbin_num);
    int min_colbin = getmin(colbin_data,colbin_num);
    int min_allbin = getmin(allbin_data,allbin_num);

    int max_rowbin = getmax(rowbin_data,rowbin_num);
    int max_colbin = getmax(colbin_data,colbin_num);
    int max_allbin = getmax(allbin_data,allbin_num);

    int ne_rowbin = getne(rowbin_data,rowbin_num);
    int ne_colbin = getne(colbin_data,colbin_num);
    int ne_allbin = getne(allbin_data,allbin_num);

    double gini_rowbin = getgini(rowbin_data,rowbin_num);
    double gini_colbin = getgini(colbin_data,colbin_num);
    double gini_allbin = getgini(allbin_data,allbin_num);

    double pratio_rowbin = getpratio(rowbin_data,rowbin_num,mtx->nnz);
    double pratio_colbin = getpratio(colbin_data,colbin_num,mtx->nnz);
    double pratio_allbin = getpratio(allbin_data,allbin_num,mtx->nnz);
    

    double gini_colbin = getgini(colbin_data,colbin_num);

    //print features
	cout << "Matrix size:" << endl;
	cout << "rows:\n" << mtx->rows << "\ncols:\n" << mtx->cols << "\nnon zeros:\n" << mtx->nnz << endl;
    
	cout << "Nonzero Skew:" << endl;
    cout << "Rows:\nave:\n" << ave_row << "\nstandardDev:\n" << standardDev_row << "\nvariance:\n" << variance_row << endl;
    cout << "minnum:\n" << min_row << "\nmaxnum:\n" << max_row << "\nnenum:\n" << ne_row << endl;
    cout << "gini:\n" << gini_row << "\npratio:\n" << pratio_row << endl;
    cout << "Cols:\nave:\n" << ave_col << "\nstandardDev:\n" << standardDev_col << "\nvariance:\n" << variance_col << endl;
	cout << "minnum:\n" << min_col << "\nmaxnum:\n" << max_col << "\nnenum:\n" << ne_col << endl;
    cout << "gini:\n" << gini_col << "\npratio:\n" << pratio_col << endl;

    cout << "Nonzero Locality:" << endl;
    cout << "Row Splits:\nave:\n" << ave_rowbin << "\nstandardDev:\n" << standardDev_rowbin << "\nvariance:\n" << variance_rowbin << endl;
    cout << "minnum:\n" << min_rowbin << "\nmaxnum:\n" << max_rowbin << "\nnenum:\n" << ne_rowbin << endl;
    cout << "gini:\n" << gini_rowbin << "\npratio:\n" << pratio_rowbin << endl;
    cout << "Col Splits:\nave:\n" << ave_colbin << "\nstandardDev:\n" << standardDev_colbin << "\nvariance:\n" << variance_colbin << endl;
    cout << "minnum:\n" << min_colbin << "\nmaxnum:\n" << max_colbin << "\nnenum:\n" << ne_colbin << endl;
    cout << "gini:\n" << gini_colbin << "\npratio:\n" << pratio_colbin << endl;
    cout << "Blocks:\nave:\n" << ave_allbin << "\nstandardDev:\n" << standardDev_allbin << "\nvariance:\n" << variance_allbin << endl;
    cout << "minnum:\n" << min_allbin << "\nmaxnum:\n" << max_allbin << "\nnenum:\n" << ne_allbin << endl;
    cout << "gini:\n" << gini_allbin << "\npratio:\n" << pratio_allbin << endl;
    
    cout << "colbin_gini:\n" << gini_colbin << endl;
    free(row_data);
    free(col_data);
    free(rowtile_data);
    free(alltile_data);
    free(rowbin_data);
    free(colbin_data);
    free(allbin_data);
    */
       // get the nonzero num in average row and col
    int* row_data = (int*)malloc(sizeof(int) * mtx->rows);
    memset(row_data, 0, sizeof(int) * mtx->rows);
    for(int i = 0; i < mtx->nnz; i++){
        row_data[mtx->row[i]]++;
    }

    //divide the matrix into K * K blocks, and get the nonzero num
    int colbin_num = (mtx->cols - 1) / K + 1;
	int* colbin_data = (int*)malloc(sizeof(int) * colbin_num);
    memset(colbin_data, 0, sizeof(int) * colbin_num);
    for(int i = 0; i < mtx->nnz; i++){
        int colbin_id = mtx->col[i] / K;
        colbin_data[colbin_id]++;
    }

    double ave_row = mtx->nnz * 1.0 / mtx->rows;
    double variance_row = getvariance(row_data,ave_row,mtx->rows);
    double dis = getdis(mtx);
    double gini_colbin = getgini(colbin_data,colbin_num);

    //print features
	cout << "Rows:\n" << mtx->rows << "\nNon zeros:\n" << mtx->nnz << endl;
    
    cout << "Ave_R:\n" << ave_row << endl;
    cout << "Dis\n" << dis << endl; 
    cout << "Gini_cb\n" << gini_colbin << endl;
    cout <<  "Aariance:\n" << variance_row << endl;
    /*
    //batch128
    double performance = -7.78282 + 0.02490 * mtx->rows -0.00125 * mtx->nnz + 4.61384 * ave_row -0.00796* dis
                        -4.93556* gini_colbin -0.00765 * variance_row;
    double pretime =  2 * mtx->nnz / performance / 1e9;
    cout << "pre_performance_128:\n" << performance << endl;
    cout << "pretime_128:\n" << pretime <<endl;
    double dtoh = 0.009+ 0.0000000784259 * mtx->rows * 8 * 512 + 0.01 * 3;
    double pipeline =(dtoh + 0.009 + 8 * 512 * mtx->rows / 4 * 0.0000000717816 + pretime * 1000 * 512 / 2) / 1000 / 512;
    cout << "dtoh_128:\n" << dtoh << endl;
    cout << "pipeline_128:\n" << pipeline <<endl;

    //batch256
    performance = 19.92324 + 0.01939 * mtx->rows -0.00100 * mtx->nnz + 3.72022 * ave_row -0.01751* dis
                        -7.05089* gini_colbin -0.01182 * variance_row;
    pretime =  2 * mtx->nnz / performance / 1e9;
    cout << "pre_performance_256:\n" << performance << endl;
    cout << "pretime_256:\n" << pretime <<endl;
    dtoh = 0.009+ 0.0000000784259 * mtx->rows * 8 * 1024 + 0.01 * 3;
    pipeline =(dtoh + 0.009 + 8 * 1024 * mtx->rows / 4 * 0.0000000717816 + pretime * 1000 * 1024 / 2) / 1000 / 1024;
    cout << "dtoh_256:\n" << dtoh << endl;
    cout << "pipeline_256:\n" << pipeline <<endl;
    
    //batch512
    performance = 48.67660 + 0.01178 * mtx->rows -0.00056 * mtx->nnz + 2.20412 * ave_row -0.01647* dis
                        -5.87084* gini_colbin -0.00899 * variance_row;
    pretime =  2 * mtx->nnz / performance / 1e9;
    cout << "pre_performance_512:\n" << performance << endl;
    cout << "pretime_512:\n" << pretime <<endl;
    dtoh = 0.009+ 0.0000000784259 * mtx->rows * 8 * 2048 + 0.01 * 3;
    pipeline =(dtoh + 0.009 + 8 * 2048 * mtx->rows / 4 * 0.0000000717816 + pretime * 1000 * 2048 / 2) / 1000 / 2048;
    cout << "dtoh_512:\n" << dtoh << endl;
    cout << "pipeline_512:\n" << pipeline <<endl;

    //batch1024
    performance = 45.71256 + 0.01268 * mtx->rows -0.00072 * mtx->nnz + 2.39320 * ave_row -0.01659 * dis
                        + 13.23380* gini_colbin + 0.00786 * variance_row;
    pretime =  2 * mtx->nnz / performance / 1e9;
    cout << "pre_performance_1024:\n" << performance << endl;
    cout << "pretime_1024:\n" << pretime <<endl;
    dtoh = 0.009+ 0.0000000784259 * mtx->rows * 8 * 4096 + 0.01 * 3;
    pipeline =(dtoh + 0.009 + 8 * 4096 * mtx->rows / 4 * 0.0000000717816 + pretime * 1000 * 4096 / 2) / 1000 / 4096;
    cout << "dtoh_1024:\n" << dtoh << endl;
    cout << "pipeline_1024:\n" << pipeline <<endl;
    */
    free(row_data);
    free(colbin_data);

}
#endif