#include <string>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <numeric>
#include <atomic>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <queue>
#include <algorithm>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("%s [matrix_num] [imbalance_ratio]\n", argv[0]);
        return 0;
    }
    string pm;
    double imbalance_ratio;
    pm = to_string(atoi(argv[1]));
    imbalance_ratio = atof(argv[2]);

    cout << "matrix_num: " << pm << endl;
    cout << "imbalance_ratio: " << imbalance_ratio << endl;

    int clusternum;
    clusternum = 6; //stoi(argv[1]);
    vector<int> cdomainsoffset(clusternum);
    vector<vector<vector<int> > > cdomainsneighs(clusternum);
    int totalsize;
    for (int c = 0; c < clusternum; ++c)
    {
        string ifilename = "dual_" + to_string(c);
        ifstream in(ifilename);
        in >> cdomainsoffset[c];
        int dnum;
        in >> dnum;
        cdomainsneighs[c].resize(dnum);
        in >> totalsize;
        for (int d = 0; d < dnum; ++d)
        {
            int num;
            in >> num;
            cdomainsneighs[c][d].resize(num);
            for (int i = 0; i < num; ++i)
            {
                in >> cdomainsneighs[c][d][i];
            }
        }
    }

    int nvtxs = totalsize;
    vector<double> vwgt(nvtxs); // TODO: 导入权重
    {
        string ifilename = pm + "pre.csv";
        ifstream in(ifilename);
        in >> totalsize;
        for (int i = 0; i < nvtxs; ++i)
        {
            in >> vwgt[i];
        }
    }

    cout << "clusternum: " << clusternum << endl;
    cout << "totalsize: " << totalsize << endl;

    vector<int> xadj(nvtxs + 1);
    vector<int> adjncy;
    int t = 1;
    for (int c = 0; c < clusternum; ++c)
    {
        for (int d = 0; d < cdomainsneighs[c].size(); ++d)
        {
            xadj[t] = xadj[t - 1] + cdomainsneighs[c][d].size();
            for (int i = 0; i < cdomainsneighs[c][d].size(); ++i) adjncy.push_back(cdomainsneighs[c][d][i]);
            t++;
        }
    }

    int nparts = clusternum;
    vector<int> part(nvtxs);
    for (int c = 0; c < clusternum; ++c)
    {
        int dnum = cdomainsneighs[c].size();
        int offset = cdomainsoffset[c];
        for (int d = offset; d < offset + dnum; ++d) part[d] = c;
    }
    vector<unordered_map<int, unordered_set<int> > > partBoundary(nparts);

    // ------------------------------------------------------------
    // 都维护一个可变动边界
    for (int i = 0; i < nvtxs; ++i)
    {
        int vert_part = part[i];
        int flag = 0;
        for (int j = xadj[i]; j < xadj[i + 1]; ++j)
        {
            int neigh_vert_id = adjncy[j];
            int neigh_vert_part = part[neigh_vert_id];
            if (neigh_vert_part != vert_part)
            {
                if (!partBoundary[vert_part].count(neigh_vert_part))
                {
                    partBoundary[vert_part][neigh_vert_part] = unordered_set<int>();
                }
                partBoundary[vert_part][neigh_vert_part].insert(i); 
                // cout << "vert_part: " << "" << vert_part << "," << neigh_vert_part << endl;
            }
        }
    }

    vector<double> pwgt(nparts);
    for (int i = 0; i < nvtxs; ++i)
    {
        pwgt[part[i]] += vwgt[i];
    }
    cout << "pwt: " << endl;
    for (int i = 0; i < nparts; ++i) cout << pwgt[i] << endl;


    double avgpw = 0; // 平均区域权值
    for (int i = 0; i < nvtxs; ++i)
    {
        avgpw += vwgt[i];
    }
    cout << "totalw: " << avgpw << endl;
    avgpw = avgpw / nparts;
    avgpw *= imbalance_ratio; // 允许一定的负载均衡误差
    int iter = 0;
    int maxIter = 10;

    for (; iter < maxIter; ++iter)
    {
        int flag = 0;
        for (int i = 0; i < nparts; ++i)
        {
            if (pwgt[i] > avgpw) flag = 1;
        }
        if (flag == 0) break;

        vector<int> pvis(nparts);
        while (1)
        {
            int pnow = -1; // 现在调整的区域 ID
            double maxpw = -1e9;
            for (int i = 0; i < nparts; ++i)
            {
                if (!pvis[i] && pwgt[i] > avgpw && pwgt[i] > maxpw)
                {
                    maxpw = pwgt[i];
                    pnow = i;
                }
            }
            if (pnow == -1) break;

            vector<int> pneigh;
            for (auto iter: partBoundary[pnow])
            {
                pneigh.push_back(iter.first);
            }

            while (pwgt[pnow] > avgpw)
            {
                int targetp = -1;
                double minw = 1e9;
                for (auto nei: pneigh)
                {
                    if (!pvis[nei] && pwgt[nei] < minw && partBoundary[pnow][nei].size())
                    {
                        minw = pwgt[nei];
                        targetp = nei;
                    }
                }
                if (targetp == -1) break;

                // boundary update
                // 选取一个进行负载均衡
                int k = rand() % (partBoundary[pnow][targetp].size());
                auto iter = partBoundary[pnow][targetp].begin();
                for (int i = 0; i < k; ++i) iter++;

                int id = *iter;
                for (int j = xadj[id]; j < xadj[id + 1]; ++j)
                {
                    int neigh_vert_id = adjncy[j];
                    if (part[neigh_vert_id] == part[id]) partBoundary[pnow][targetp].insert(neigh_vert_id);
                }
                // cout << "vid: " << id << " " << pnow << " -> " << targetp << ", oripart: " << part[id] << endl;
                part[id] = targetp;
                partBoundary[pnow][targetp].erase(id);
                partBoundary[targetp][pnow].insert(id);
                pwgt[pnow] -= vwgt[id];
                pwgt[targetp] += vwgt[id];
                // for (int i = 0; i < clusternum; ++i) cout << pwgt[i] << " ";
                // cout << endl;
                // for (int i = 0; i < clusternum; ++i) cout << pwgt[i] << " ";
                // cout << endl;
                // vector<double> pvwgt(clusternum);
                // for (int i = 0; i < totalsize; ++i) 
                // {
                //     pvwgt[part[i]] += vwgt[i];
                // }
                // for (int i = 0; i < clusternum; ++i) cout << pvwgt[i] << " ";
                // cout << endl;
                // cout << endl;
            }

            pvis[pnow] = 1;
        }
    }

    cout << "########### after balance pwt" << endl;
    for (int i = 0; i < nparts; ++i) 
    {
        cout << pwgt[i] << endl;
    }

    {
        // cout << "########### after balance pvwt" << endl;
        // vector<double> pvwgt(clusternum);
        // for (int i = 0; i < totalsize; ++i) 
        // {
        //     pvwgt[part[i]] += vwgt[i];
        // }
        // for (int i = 0; i < clusternum; ++i)
        // {
        //     cout << pvwgt[i] << endl;
        // }
        // cout << "########### after balance pnwt" << endl;

    }
    {
        string ofilename = "matrix_partition_" + pm;
        ofstream out(ofilename);
        out << nvtxs << endl;
        for (int i = 0; i < nvtxs; ++i)
        {
            out << part[i] << endl;
        }
        out.close();

        cout << "########### check matrix pwt" << endl;
        string ifilename = "matrix_partition_" + pm;
        ifstream in(ifilename);
        in >> totalsize;
        vector<double> pnwgt(clusternum);
        for (int i = 0; i < totalsize; ++i)
        {
            int x;
            in >> x;
            pnwgt[x] += vwgt[i];
        }
        in.close();
        for (int i = 0; i < clusternum; ++i)
        {
            cout << pnwgt[i] << endl;
        }
    }


    return 0;
}
