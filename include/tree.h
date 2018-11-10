#pragma once
#include <vector>
#include <string>
using namespace std;


namespace xgboost {
	class Tree {
	public:
		Tree();
		~Tree();
		float predict_leaf_value(const vector<float>& dataset_one);
		string describe_tree();

		int split_feature;
		float split_value;
		float split_gain;
		float internal_value;
		float leaf_value;
		Tree* tree_left;
		Tree* tree_right;
	};
}
