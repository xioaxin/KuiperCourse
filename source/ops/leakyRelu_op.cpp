//
// Created by zpx on 2023/01/01.
//
#include "ops/leakyRelu_op.h"

namespace kuiper_infer {
    LeakyReluOperator::LeakyReluOperator(float thresh) : thresh_(thresh), Operator(OpType::kOperatorLeakyRelu) {};

    void LeakyReluOperator::set_thresh(const float thresh) { thresh_ = thresh; }

    float LeakyReluOperator::get_thresh() const { return thresh_; }
}