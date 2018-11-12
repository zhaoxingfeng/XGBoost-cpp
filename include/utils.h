#pragma once
#include <vector>
using namespace std;


float calculate_auc(vector<float>& labels, vector<float>& pvalues);
float calculate_ks(vector<float>& labels, vector<float>& pvalues);
float calculate_acc(vector<float>& labels, vector<float>& pvalues);

