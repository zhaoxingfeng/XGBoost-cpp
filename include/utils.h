#pragma once
#include <vector>


float calculate_auc(std::vector<float>& labels, std::vector<float>& pvalues);
float calculate_ks(std::vector<float>& labels, std::vector<float>& pvalues);
float calculate_acc(std::vector<float>& labels, std::vector<float>& pvalues);

