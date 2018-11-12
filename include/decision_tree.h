#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "config.h"
#include "pandas.h"
#include "tree.h"
using namespace std;


namespace xgboost {
	struct BestSplitInfo {
		int best_split_feature;
		float best_split_value;
		float best_split_gain;
		float best_internal_value;
		vector<int> best_sub_dataset_left;
		vector<int> best_sub_dataset_right;
	};

	class BaseDecisionTree {
	public:
		BaseDecisionTree(Config conf);
		~BaseDecisionTree() = default;
		Tree* fit(const vector<vector<float>>& features_in, const vector<float>& labels_in,
			const vector<float>& grad_in, const vector<float>& hess_in);

	private:
		const Config config;
		Tree* decision_tree;
		vector<vector<float>> features;
		vector<float> labels;
		vector<float> grad;
		vector<float> hess;

		Tree* _fit(vector<int>& sub_dataset, int depth);
		BestSplitInfo choose_best_feature(const vector<int>& sub_dataset);
		float calculate_leaf_value(const vector<int>& sub_dataset);
		float calculate_split_gain(const float& left_grad_sum, const float& left_hess_sum,
			const float& right_grad_sum, const float& right_hess_sum);
	};
}
