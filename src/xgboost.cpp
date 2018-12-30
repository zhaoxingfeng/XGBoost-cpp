#include <numeric>
#include <algorithm>
#include "xgboost.h"
#include "decision_tree.h"
#include "tree.h"
using namespace std;


namespace xgboost {
	XGBoost::XGBoost(Config conf) :config(conf) {};
	XGBoost::~XGBoost() {};

	//训练模型主函数入口
	void XGBoost::fit(const vector<vector<float>>& features, const vector<float>& labels) {
		float mean = accumulate(labels.begin(), labels.end(), 0) / (float)labels.size();
		pred_0 = 0.5 * log((1 + mean) / (1 - mean));

		Gradients gradients;
		for (int i = 0; i < labels.size(); ++i) {
			gradients = calculate_grad_hess(labels[i], pred_0);
			grad.push_back(gradients.grad);
			hess.push_back(gradients.hess);
		}

		for (int stage = 1; stage <= config.n_estimators; ++stage) {
			cout << "=============================== iter: " << stage << " ===============================" << endl;
			Tree* tree_stage;
			BaseDecisionTree base_decision_tree = BaseDecisionTree(config);
			tree_stage = base_decision_tree.fit(features, labels, grad, hess);
			trees.push_back(tree_stage);
			cout << tree_stage->describe_tree() << endl;

			for (int i = 0; i < labels.size(); ++i) {
				float y_pred = tree_stage->predict_leaf_value(features[i]);
				gradients = calculate_grad_hess(labels[i], y_pred);
				grad[i] = grad[i] + config.learning_rate * gradients.grad;
				hess[i] = hess[i] + config.learning_rate * gradients.hess;
			}
		}
	}

	//计算一阶和二阶导数
	Gradients XGBoost::calculate_grad_hess(float y, float y_pred) {
		Gradients gradients;
		float pred = 1.0 / (1.0 + exp(-y_pred));
		float grad = (-y + (1 - y) * exp(pred)) / (1 + exp(pred));
		float hess = exp(pred) / pow((1 + exp(pred)), 2);
		gradients = { grad, hess };
		return gradients;
	}

	//给定样本特征，预测p值
	vector<float> XGBoost::predict_proba(const vector<float>& features) {
		float pred = pred_0;
		vector<float> res;
		float p_0;
		for (Tree* tree : trees) {
			pred += config.learning_rate * tree->predict_leaf_value(features);

		}
		p_0 = 1.0 / (1 + exp(2 * pred));
		res.push_back(p_0);
		res.push_back(1 - p_0);
		return res;
	}
}
