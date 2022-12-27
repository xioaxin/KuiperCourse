//
// Created by zpx on 2022/12/26.
//
#include "ops/relu_op.h"

namespace kuiper_infer {
    ReluOperator::ReluOperator(float thresh) : thresh_(thresh), Operator(OpType::kOperatorRelu) {};

    void ReluOperator::set_thresh(float thresh) {
        thresh_ = thresh;
    }

    float ReluOperator::get_thresh() const { return thresh_; }
}