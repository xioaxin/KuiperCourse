//
// Created by zpx on 2023/01/01.
//
#include "ops/relu6_op.h"
#include "ops/ops.h"

namespace kuiper_infer {
    Relu6Operator::Relu6Operator(float thresh) : thresh_(thresh), Operator(OpType::kOperatorRelu6) {};

    void Relu6Operator::set_thresh(float thresh) {
        thresh_ = thresh;
    }

    float Relu6Operator::get_thresh() const { return thresh_; }
}