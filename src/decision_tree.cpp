#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "config.h"
#include "pandas.h"
#include "decision_tree.h"


namespace xgboost {
	BaseDecisionTree::BaseDecisionTree(Config conf) :config(conf) {};
	//BaseDecisionTree::~BaseDecisionTree() {};

	//训练单棵回归树
	Tree* BaseDecisionTree::fit(const vector<vector<float>>& features_in, const vector<float>& labels_in,
			const vector<float>& grad_in, const vector<float>& hess_in) {
		features = features_in;
		labels = labels_in;
		grad = grad_in;
		hess = hess_in;
		vector<int> sub_dataset;
		for (int i = 0; i < labels.size(); ++i) {
			sub_dataset.push_back(i);
		}
		decision_tree = _fit(sub_dataset, 1);
		return decision_tree;
	}

	//递归生成树
	Tree* BaseDecisionTree::_fit(vector<int>& sub_dataset, int depth) {
		//计算二阶导数之和
		float sub_hess = 0.0;
		for (int i : sub_dataset) {
			sub_hess += hess[i];
		}

		//当前节点样本数量小于最小叶子节点样本数量或二阶导数之和小于给定weight，则停止分裂
		if (sub_dataset.size() <= config.min_data_in_leaf || sub_hess <= config.min_child_weight) {
			Tree* tree = new Tree();
			tree->leaf_value = calculate_leaf_value(sub_dataset);
			return tree;
		}

		if (depth <= config.max_depth) {
			BestSplitInfo best_split_info = choose_best_feature(sub_dataset);
			Tree* tree = new Tree();

			if (best_split_info.best_sub_dataset_left.size() < config.min_data_in_leaf ||
				best_split_info.best_sub_dataset_right.size() < config.min_data_in_leaf) {
				tree->leaf_value = calculate_leaf_value(sub_dataset);
				return tree;
			}
			else {
				tree->split_feature = best_split_info.best_split_feature;
				tree->split_value = best_split_info.best_split_value;
				tree->split_gain = best_split_info.best_split_gain;
				tree->internal_value = best_split_info.best_internal_value;
				tree->tree_left = _fit(best_split_info.best_sub_dataset_left, depth + 1);
				tree->tree_right = _fit(best_split_info.best_sub_dataset_right, depth + 1);
				return tree;
			}
		}
		else {
			Tree* tree = new Tree();
			tree->leaf_value = calculate_leaf_value(sub_dataset);
			return tree;
		}
	}

	//寻找最优分割特征和分割点
	BestSplitInfo BaseDecisionTree::choose_best_feature(const vector<int>& sub_dataset) {
		int best_split_feature;
		float best_split_value;
		float best_split_gain = -1e10;
		float best_internal_value;
		vector<int> best_sub_dataset_left;
		vector<int> best_sub_dataset_right;

		vector<int> sub_dataset_left;
		vector<int> sub_dataset_right;
		float left_grad_sum;
		float left_hess_sum;
		float right_grad_sum;
		float right_hess_sum;
		
		for (int i = 0; i < features[0].size(); ++i) {
			//找到该特征下所有可能的分割点
			vector<float> unique_values;
			unique_values.push_back(features[0][i]);
			for (int j = 0; j < sub_dataset.size(); ++j) {
				if (find(unique_values.begin(), unique_values.end(), features[j][i]) == unique_values.end()) {
					unique_values.push_back(features[j][i]);
				}
			}

			//贪婪搜索，寻找使gain最大的分割点
			for (float split_value : unique_values) {
				sub_dataset_left.clear();
				sub_dataset_right.clear();
				left_grad_sum = 0;
				left_hess_sum = 0;
				right_grad_sum = 0;
				right_hess_sum = 0;

				for (int index : sub_dataset) {
					if (features[index][i] <= split_value) {
						sub_dataset_left.push_back(index);
						left_grad_sum += grad[index];
						left_hess_sum += hess[index];
					}
					else {
						sub_dataset_right.push_back(index);
						right_grad_sum += grad[index];
						right_hess_sum += hess[index];
					}
				}
				float split_gain = calculate_split_gain(left_grad_sum, left_hess_sum, right_grad_sum, right_hess_sum);
				if (best_split_gain < split_gain) {
					best_split_feature = i;
					best_split_value = split_value;
					best_split_gain = split_gain;
					best_sub_dataset_left = sub_dataset_left;
					best_sub_dataset_right = sub_dataset_right;
				}
			}
		}
		best_internal_value = calculate_leaf_value(sub_dataset);
		BestSplitInfo best_split_info;
		best_split_info = { best_split_feature, best_split_value, best_split_gain, best_internal_value, best_sub_dataset_left, best_sub_dataset_right };
		return best_split_info;
	}

	//计算分裂后的增益
	float BaseDecisionTree::calculate_split_gain(const float& left_grad_sum, const float& left_hess_sum,
		const float& right_grad_sum, const float& right_hess_sum) {
		float tmp1 = pow(left_grad_sum, 2) / (left_hess_sum + config.reg_lambda);
		float tmp2 = pow(right_grad_sum, 2) / (right_hess_sum + config.reg_lambda);
		float tmp3 = pow((left_grad_sum + right_grad_sum), 2) / (left_hess_sum + right_hess_sum + config.reg_lambda);
		return (tmp1 + tmp2 - tmp3) / 2 - config.reg_gamma;
	}

	//计算叶子节点取值
	float BaseDecisionTree::calculate_leaf_value(const vector<int>& sub_dataset) {
		float grad_sum = 0;
		float hess_sum = 0;
		for (int index : sub_dataset) {
			grad_sum += grad[index];
			hess_sum += hess[index];
		}
		return -grad_sum / (hess_sum + config.reg_lambda);
	}
}
