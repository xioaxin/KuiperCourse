//
// Created by zpx on 2023/02/25.
//
#include "ops/cat_op.h"

namespace kuiper_infer {
    CatOperator::CatOperator() : RuntimeOperator(OpType::kOperatorCat) {}

    CatOperator::CatOperator(uint32_t dim) : RuntimeOperator(OpType::kOperatorCat), dim_(dim) {}

    const uint32_t CatOperator::getDim() const {
        return dim_;
    }

    void CatOperator::setDim(const uint32_t dim) {
        this->dim_ = dim;
    }

    void CatOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
    }

    void CatOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> CatOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorCat);
        std::shared_ptr<RuntimeOperator> catOperator = std::make_shared<CatOperator>();
        return catOperator;
    }

    RuntimeOperatorRegistererWrapper kCatOperator(OpType::kOperatorCat, CatOperator::CreateInstance);
}