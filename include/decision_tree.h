#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "config.h"
#include "pandas.h"
#include "tree.h"


namespace xgboost {
	struct BestSplitInfo {
		int best_split_feature = 0;
		float best_split_value = 0;
		float best_split_gain = -1e10;
		float best_internal_value = 0;
		std::vector<int> best_sub_dataset_left;
		std::vector<int> best_sub_dataset_right;
	};

	class BaseDecisionTree {
	public:
		BaseDecisionTree(Config conf);
		~BaseDecisionTree() = default;
		Tree* fit(const std::vector<std::vector<float>>& features_in, const std::vector<float>& labels_in,
			const std::vector<float>& grad_in, const std::vector<float>& hess_in);

	private:
		const Config config;
		Tree* decision_tree;
		std::vector<std::vector<float>> features;
		std::vector<float> labels;
		std::vector<float> grad;
		std::vector<float> hess;

		Tree* _fit(std::vector<int>& sub_dataset, int depth);
		BestSplitInfo choose_best_split_feature(const std::vector<int>& sub_dataset);
		BestSplitInfo choose_best_split_value(const std::vector<int>& sub_dataset, int feature_index);
		float calculate_leaf_value(const std::vector<int>& sub_dataset);
		float calculate_split_gain(const float& left_grad_sum, const float& left_hess_sum,
			const float& right_grad_sum, const float& right_hess_sum);
	};
}
