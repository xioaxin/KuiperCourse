//
// Created by zpx on 2023/02/26.
//
#include "ops/flatten_op.h"

namespace kuiper_infer {
    FlattenOperator::FlattenOperator(uint32_t start_dim, uint32_t end_dim) : Operator(OpType::kOperatorFlatten),
                                                                             start_dim_(start_dim), end_dim_(end_dim) {}

    const uint32_t FlattenOperator::getEndDim() const {
        return end_dim_;
    }

    const uint32_t FlattenOperator::getStartDim() const {
        return start_dim_;
    }

    void FlattenOperator::setStartDim(const uint32_t start_dim) {
        this->start_dim_ = start_dim;
    }

    void FlattenOperator::setEndDim(const uint32_t end_dim) {
        this->end_dim_ = end_dim;
    }
}