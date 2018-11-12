#include <iostream>
#include <vector>
#include <time.h>
#include "c_api.h"
#include "pandas.h"
#include "xgboost.h"
#include "pandas.h"
using namespace std;
using namespace pandas;
using namespace xgboost;


#define API_BEGIN() try {
#define API_END() } catch(std::runtime_error &_except_) { return -1; } return 0;

//训练模型接口
XGB_DLL int Train(Config* conf, const float *data, const float *label, int nrow, int ncol, BoosterHandle *out) {
	API_BEGIN();
	vector< vector<float> > features(nrow, vector<float>(ncol));
	vector<float> labels;

	for (int i = 0; i < ncol; ++i, data += nrow) {
		for (int j = 0; j < nrow; ++j) {
			features[j][i] = data[j];
		}
	}
	for (int k = 0; k < nrow; ++k) {
		labels.push_back(label[k]);
	}
	
	XGBoost *model = new XGBoost(*conf);
	model->fit(features, labels);
	*out = model;
	API_END();
}

//模型预测接口
XGB_DLL int Predict(const float *data, int nrow, int ncol, BoosterHandle *handle, float *out_result) {
	API_BEGIN();
	vector< vector<float> > features(nrow, vector<float>(ncol));

	for (int i = 0; i < ncol; ++i, data += nrow) {
		for (int j = 0; j < nrow; ++j) {
			features[j][i] = data[j];
		}
	}

	XGBoost* xgboost_model = static_cast<XGBoost*>(*handle);
	for (int i = 0; i < features.size(); ++i) {
		out_result[i] = xgboost_model->predict_proba(features[i])[1];
	}
	API_END();
}
