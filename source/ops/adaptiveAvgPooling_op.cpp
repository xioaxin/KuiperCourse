//
// Created by zpx on 2023/02/21.
//
#include "ops/adaptiveAvgPooling_op.h"

namespace kuiper_infer {
    AdaptiveAvgPoolingOperator::AdaptiveAvgPoolingOperator(uint32_t output_h, uint32_t output_w)
            : output_h_(output_h), output_w_(output_w), Operator(OpType::kOperatorAdaptiveAvgPooling) {}

    uint32_t AdaptiveAvgPoolingOperator::get_output_h() {
        return output_h_;
    }

    uint32_t AdaptiveAvgPoolingOperator::get_output_w() {
        return output_w_;
    }

    void AdaptiveAvgPoolingOperator::set_output_h(uint32_t output_h) {
        this->output_h_ = output_h;
    }

    void AdaptiveAvgPoolingOperator::set_output_w(uint32_t output_w) {
        this->output_w_ = output_w;
    }
}