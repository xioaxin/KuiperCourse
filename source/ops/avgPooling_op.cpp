//
// Created by zpx on 2023/02/18.
//
//
// Created by zpx on 2023/02/03.
//
#include "ops/avgPooling_op.h"

namespace kuiper_infer {
    AvgPoolingOperator::AvgPoolingOperator(uint32_t pooling_h, uint32_t pooling_w, uint32_t stride_h, uint32_t stride_w,
                                           uint32_t padding_h, uint32_t padding_w) :
            pooling_h_(pooling_h), pooling_w_(pooling_w), stride_h_(stride_h), stride_w_(stride_w),
            padding_w_(padding_w), padding_h_(padding_h), Operator(OpType::kOperatorAvgPooling) {
    }

    void AvgPoolingOperator::set_pooling_h(uint32_t pool_h) {
        pooling_h_ = pool_h;
    }

    void AvgPoolingOperator::set_pooling_w(uint32_t pool_w) {
        pooling_w_ = pool_w;
    }

    void AvgPoolingOperator::set_padding_h(uint32_t padding_h) {
        padding_h_ = padding_h;
    }

    void AvgPoolingOperator::set_padding_w(uint32_t padding_w) {
        padding_w_ = padding_w;
    }

    void AvgPoolingOperator::set_stride_h(uint32_t stride_h) {
        stride_h_ = stride_h;
    }

    void AvgPoolingOperator::set_stride_w(uint32_t stride_w) {
        stride_w_ = stride_w;
    }

    uint32_t AvgPoolingOperator::get_padding_h() {
        return padding_h_;
    }

    uint32_t AvgPoolingOperator::get_padding_w() {
        return padding_w_;
    }

    uint32_t AvgPoolingOperator::get_pooling_h() {
        return pooling_h_;
    }

    uint32_t AvgPoolingOperator::get_pooling_w() {
        return pooling_w_;
    }

    uint32_t AvgPoolingOperator::get_stride_h() {
        return stride_h_;
    }

    uint32_t AvgPoolingOperator::get_stride_w() {
        return stride_w_;
    }
}