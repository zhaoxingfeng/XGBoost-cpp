#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include "config.h"
#include "pandas.h"
#include "xgboost.h"
#include "tree.h"
#include "utils.h"
#include "numpy.h"
#include <list>
using namespace std;
using namespace xgboost;
using namespace pandas;
using namespace numpy;


int main() {
	clock_t startTime, endTime;
	startTime = clock();

	//设置模型超参
	Config config;
	config.n_estimators = 50;
	config.learning_rate = 0.1;
	config.max_depth = 6;
	config.min_samples_split = 50;
	config.min_data_in_leaf = 20;
	config.reg_gamma = 0.3;
	config.reg_lambda = 0.3;
	config.colsample_bytree = 0.8;
	config.min_child_weight = 5;
	config.max_bin = 100;

	/*Config config;
	config.n_estimators = 5;
	config.learning_rate = 0.4;
	config.max_depth = 6;
	config.min_samples_split = 500;
	config.min_data_in_leaf = 100;
	config.reg_gamma = 0.3;
	config.reg_lambda = 0.3;
	config.colsample_bytree = 1.0;
	config.min_child_weight = 5;
	config.max_bin = 100;*/

	//读取样本
	pandas::Dataset dataset = pandas::read_csv("./source/pima indians.csv", ',', -1);
	//pandas::Dataset dataset = pandas::read_csv("./source/credit_card.csv", ',', -1, 5000);
	XGBoost xgboost = XGBoost(config);
	xgboost.fit(dataset.features, dataset.labels);

	vector<float> pvalues;
	for (int i = 0; i < dataset.labels.size(); ++i) {
		pvalues.push_back(xgboost.predict_proba(dataset.features[i])[1]);
	}
	//计算模型准确率
	cout << "ACC: " << calculate_acc(dataset.labels, pvalues) << endl;
	pandas::save_csv(pvalues, "test_p.csv");

	endTime = clock();
	cout << "Totle Time : " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	system("pause");
	return 0;
}
