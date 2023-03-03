<<<<<<< HEAD
#include <glog/logging.h>
#include "iostream"
#include <armadillo>
#include <benchmark/benchmark.h>
using namespace std;
using namespace arma;

int main() {
    vec x = regspace(1, 4);
    for (auto item: x)cout << item <<" ";
    cout << endl;
    vec y = square(x);
    for (auto item: y)cout << item <<" ";
    cout << endl;
    vec xx = regspace(x.min(), 0.5, x.max());
    for (auto item: xx)cout << item <<" ";
    cout << endl;
    vec yy;
    interp1(x, y, xx, yy);  // use linear interpolation by default
    for (auto item: yy)cout << item <<" ";
    cout << endl;
    return 0;
=======
#include <iostream>
#include <armadillo>
int main() {
  arma::fmat in_1(32, 32, arma::fill::ones);
  arma::fmat in_2(32, 32, arma::fill::ones);

  arma::fmat out = in_1 + in_2;
  std::cout << "rows " << out.n_rows << "\n";
  std::cout << "cols " << out.n_cols << "\n";
  std::cout << "value " << out.at(0) << "\n";
  return 0;
>>>>>>> cc18220129f36e4521c5895c1dab566b2107d767
}
