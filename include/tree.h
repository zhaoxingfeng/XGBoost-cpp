#pragma once
#include <vector>
#include <string>


namespace xgboost {
	class Tree {
	public:
		Tree();
		~Tree() = default;
		float predict_leaf_value(const std::vector<float>& dataset_one);
		std::string describe_tree();

		int split_feature;
		float split_value;
		float split_gain;
		float internal_value;
		float leaf_value;
		Tree* tree_left;
		Tree* tree_right;
	};
}
