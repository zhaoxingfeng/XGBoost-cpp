#pragma once
#include <vector>
#include "decision_tree.h"
#include "tree.h"
#include "config.h"


namespace xgboost {
	struct Gradients {
		float grad;
		float hess;
	};

	class XGBoost {
	public:
		XGBoost(Config conf);
		~XGBoost();
		void fit(const std::vector<std::vector<float>>& features, const std::vector<float>& labels);
		std::vector<float> predict_proba(const std::vector<float>& features);
		const Config config;
		float pred_0;

	private:
		std::vector<Tree*> trees;
		std::vector<float> grad;
		std::vector<float> hess;
		Gradients calculate_grad_hess(float y, float y_pred);
	};
}
