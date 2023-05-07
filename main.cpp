#include <glog/logging.h>
#include "iostream"
#include <armadillo>
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace arma;

int main() {
    mat A(5, 5, fill::zeros);
    mat B = arma::exp(A);
    for (int i = 0; i < B.n_rows; ++i) {
        for (int j = 0; j < B.n_cols; ++j) {
            cout << B.at(i, j) << " ";
        }
        cout << endl;
    }
    return 0;
}
