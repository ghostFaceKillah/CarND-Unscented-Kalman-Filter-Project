#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse.fill(0.0);

    size_t N = estimations.size();
    for (size_t i = 0; i < N; ++i) {
        VectorXd diff = estimations[i] - ground_truth[i];
        VectorXd diff_square = diff.array() * diff.array();
        rmse = rmse + diff_square;
    }

    rmse /= estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}
