#pragma once
#include <string>
#include <vector>


namespace pandas {
	struct Dataset {
		std::vector<std::vector<float>> features;
		std::vector<float> labels;
	};

	Dataset read_csv(std::string file_path, char sep, float fillna, int n_rows = 1000000);
	void save_csv(const std::vector<float>& dataset_vect, const std::string file_path);
}