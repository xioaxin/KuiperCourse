#include <glog/logging.h>
#include "iostream"
#include <armadillo>

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
}
