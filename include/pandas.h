#pragma once
#include <string>
#include <vector>
using namespace std;


namespace pandas {
	struct Dataset {
		vector<vector<float>> features;
		vector<float> labels;
	};

	Dataset read_csv(char* file_path, char sep, float fillna);
	void save_csv(const vector<float>& dataset_vect, const string path);
}