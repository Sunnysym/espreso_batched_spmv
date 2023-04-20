#include <vector>
#include <algorithm>
#include <cstdio>

#include "DPMap.h"

std::vector<double> CoresWeight;
std::vector<double> JobsWeight;
std::vector<std::vector<int>> DPMap;

void processor_bipartition(
const std::vector<int> &D, 
std::vector<int> &D0, std::vector<int> &D1
) {
    double sum = 0;
    for (int i = 0; i < D.size(); ++i) {
        sum += CoresWeight[D[i]];
    }

    double half = sum / 2.;
    double cter = 0;
    int forwardBack = 0;
    int i = 0, j = D.size() - 1;
    while (i < j) {
        if (forwardBack == 0) {
            cter += CoresWeight[D[i]];
            i++;
        } else {
            cter += CoresWeight[D[j]];
            j--;
        }
        if (cter >= half) break;
        forwardBack = !forwardBack;
    }

    for (int k = 0; k < i; ++k) {
        D0.push_back(D[k]);
    }
    for (int k = i; k <= j; ++k) {
        D1.push_back(D[k]);
    }
    for (int k = j + 1; k < D.size(); ++k) {
        D0.push_back(D[k]);
    }
    
    return ;
}

void process_bipartition(
const std::vector<int> &P, 
const std::vector<int> &D0, const std::vector<int> &D1,
std::vector<int> &P0, std::vector<int> &P1
) {
    double D0sum = 0, D1sum = 0;
    for (int i = 0; i < D0.size(); ++i) {
        D0sum += CoresWeight[D0[i]];
    }
    for (int i = 0; i < D1.size(); ++i) {
        D1sum += CoresWeight[D1[i]];
    }
    double Psum = 0;
    for (int i = 0; i < P.size(); ++i) {
        Psum += JobsWeight[P[i]];
    }

    double half = Psum * D0sum / (D0sum + D1sum);
    double cter = 0;
    int forwardBack = 0;
    int i = 0, j = P.size() - 1;
    while (i < j) {
        if (forwardBack == 0) {
            cter += JobsWeight[P[i]];
            i++;
        } else {
            cter += JobsWeight[P[j]];
            j--;
        }
        if (cter >= half) break;
        forwardBack = !forwardBack;
    }

    for (int k = 0; k < i; ++k) {
        P0.push_back(P[k]);
    }
    for (int k = i; k <= j; ++k) {
        P1.push_back(P[k]);
    }
    for (int k = j + 1; k < P.size(); ++k) {
        P0.push_back(P[k]);
    }

    return ;
}

void processMapping2(
const std::vector<int> &D,
const std::vector<int> &P
) {
    if (P.size() == 0) return ;
    if (D.size() == 1) {
        DPMap[D[0]].resize(P.size());
        for (int i = 0; i < P.size(); ++i) {
            DPMap[D[0]][i] = P[i];
        }
        return ;
    }

    std::vector<int> D0, D1;
    processor_bipartition(D, D0, D1);
    std::vector<int> P0, P1;
    process_bipartition(P, D0, D1, P0, P1);
    
    processMapping2(D0, P0);
    processMapping2(D1, P1);

    return ;
}

void processMapping(
const std::vector<double> &DInfo,
const std::vector<double> &PInfo,
std::vector<std::vector<int>> &DP
) {
    CoresWeight.clear();
    JobsWeight.clear();
    DPMap.clear();
    
    std::vector<int> P;
    std::vector<int> D;
    D.resize(DInfo.size());
    P.resize(PInfo.size());
    for (int i = 0; i < D.size(); ++i) D[i] = i;
    for (int i = 0; i < P.size(); ++i) P[i] = i;

    // Sort Dsc
    sort(D.begin(), D.end(), [&](const int &a, const int &b) {
        return DInfo[a] > DInfo[b];
    });
    sort(P.begin(), P.end(), [&](const int &a, const int &b) {
        return PInfo[a] > PInfo[b];
    });

    CoresWeight.resize(DInfo.size());
    JobsWeight.resize(PInfo.size());
    DPMap.resize(DInfo.size());
    for (int i = 0; i < D.size(); ++i) CoresWeight[i] = DInfo[i];
    for (int i = 0; i < P.size(); ++i) JobsWeight[i] = PInfo[i];
    
    processMapping2(D, P);

    DP.swap(DPMap);

    return ;
}

void printDPResult(const std::vector<double> &DInfo, const std::vector<double> &PInfo, std::vector<std::vector<int>> &DP)
{
    printf("----------------------\n");
    printf("DP result: \n");
    for (int i = 0; i < DInfo.size(); ++i) {
        int Psum = 0;
        for (int j = 0; j < DP[i].size(); ++j) {
            Psum += PInfo[DP[i][j]];
        }
        printf("%d, D: %.1f, Psize: %d, Psum: %d\n", i, DInfo[i], DP[i].size(), Psum);
    }
}