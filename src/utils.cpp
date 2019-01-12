#include <fstream>
#include "utils.h"
using namespace std;


//计算模型AUC
float calculate_auc(vector<float>& labels, vector<float>& pvalues) {
	return 0.0;
}

//计算模型KS
float calculate_ks(vector<float>& labels, vector<float>& pvalues) {
	return 0.0;
}

//计算模型准确率ACC
float calculate_acc(vector<float>& labels, vector<float>& pvalues) {
	int count_right = 0;
	for (int i = 0; i < labels.size(); ++i) {
		if ((labels[i] == 0.0 && pvalues[i] < 0.5) || (labels[i] == 1.0 && pvalues[i] >= 0.5)) {
			count_right += 1;
		}
	}
	return (float)count_right / labels.size();
}

