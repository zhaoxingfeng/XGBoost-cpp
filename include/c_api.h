#pragma once
#include <vector>
#define XGB_EXTERN_C extern "C"
#define XGB_DLL XGB_EXTERN_C __declspec(dllexport)
#include "config.h"
#include "pandas.h"
#include "xgboost.h"


typedef xgboost::XGBoost *BoosterHandle;
XGB_DLL int Train(xgboost::Config* config, const float *feature, const float *label, int nrow, int ncol, BoosterHandle *out);
XGB_DLL int Predict(const float *feature, int nrow, int ncol, BoosterHandle *handle, float *out_result);
