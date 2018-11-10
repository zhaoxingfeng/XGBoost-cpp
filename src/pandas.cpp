#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "pandas.h"
using namespace std;


namespace pandas {
	//读取csv文件，空值进行填充
	Dataset read_csv(char* file_path, char sep, float fillna) {
		Dataset dataset;
		vector<vector<float>> features;
		vector<float> labels;

		ifstream ifs(file_path);
		string line;
		while (getline(ifs, line)) {
			stringstream ss(line);
			vector<float> vect_line;
			string tmp;
			while (getline(ss, tmp, sep)) {
				if (tmp == "") {
					vect_line.push_back(fillna);
				}
				else {
					vect_line.push_back(stof(tmp));
				}
			}
			labels.push_back(vect_line.back());
			vect_line.pop_back();
			features.push_back(vect_line);
		}

		dataset = { features, labels };
		return dataset;
	}

	// 将给定的向量写到csv文件
	void save_csv(const vector<float>& dataset_vect, const string path) {
		ofstream outFile;
		outFile.open(path, ios::out);
		for (float value : dataset_vect) {
			outFile << value << endl;
		}
		outFile.close();
	}
}
