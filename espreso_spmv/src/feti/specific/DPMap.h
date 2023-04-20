#include <vector>

void processMapping(const std::vector<double> &DInfo, const std::vector<double> &PInfo, std::vector<std::vector<int>> &DP);
void printDPResult(const std::vector<double> &DInfo, const std::vector<double> &PInfo, std::vector<std::vector<int>> &DP);