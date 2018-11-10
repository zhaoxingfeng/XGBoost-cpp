#include <string>
#include <vector>
#include <iostream>
#include <time.h>
#include "config.h"
#include "pandas.h"
#include "xgboost.h"
#include "tree.h"
#include "utils.h"
using namespace std;
using namespace xgboost;
using namespace pandas;


int main() {
	clock_t startTime, endTime;
	startTime = clock();

	//设置模型超参
	Config config;
	config.n_estimators = 5;
	config.learning_rate = 0.4;
	config.max_depth = 6;
	config.min_data_in_leaf = 40;
	config.reg_gamma = 0.2;
	config.reg_lambda = 0.3;
	config.colsample_bytree = 0.8;
	config.min_child_weight = 5;

	//读取训练样本
	pandas::Dataset dataset_train = pandas::read_csv("./source/pima_indians_train.csv", ',', -1);
	XGBoost xgboost = XGBoost(config);
	xgboost.fit(dataset_train.features, dataset_train.labels);

	vector<float> pvalues_train;
	vector<float> pvalues_test;
	float pvalue;
	for (int i = 0; i < dataset_train.labels.size(); ++i) {
		pvalue = xgboost.predict_proba(dataset_train.features[i])[1];
		pvalues_train.push_back(pvalue);
	}

	//读取训练样本
	pandas::Dataset dataset_test = pandas::read_csv("./source/pima_indians_test.csv", ',', -1);
	for (int i = 0; i < dataset_test.labels.size(); ++i) {
		pvalue = xgboost.predict_proba(dataset_test.features[i])[1];
		pvalues_test.push_back(pvalue);
	}

	//计算模型准确率
	cout << "Train ACC: " << calculate_acc(dataset_train.labels, pvalues_train) << endl;
	cout << "Test ACC: " << calculate_acc(dataset_test.labels, pvalues_test) << endl;

	endTime = clock();
	cout << "Totle Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;
}
