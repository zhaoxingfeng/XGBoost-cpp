#pragma once
#include <vector>
#include "decision_tree.h"
#include "tree.h"
#include "config.h"
using namespace std;


namespace xgboost {
	struct Gradients {
		float grad;
		float hess;
	};

	class XGBoost {
	public:
		XGBoost(Config conf);
		~XGBoost();
		void fit(const vector<vector<float>>& features, const vector<float>& labels);
		vector<float> predict_proba(const vector<float>& features);
		const Config config;
		float pred_0;

	private:
		vector<Tree*> trees;
		vector<float> grad;
		vector<float> hess;
		Gradients calculate_grad_hess(float y, float y_pred);
	};
}
