//
// Created by zpx on 2023/02/25.
//
#include "ops/cat_op.h"

namespace kuiper_infer {
    CatOperator::CatOperator(uint32_t dim) : Operator(OpType::kOperatorCat), dim_(dim) {}

    const uint32_t CatOperator::getDim() const {
        return dim_;
    }

    void CatOperator::setDim(const uint32_t dim) {
        this->dim_ = dim;
    }
}