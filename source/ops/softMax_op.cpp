//
// Created by zpx on 2023/02/25.
//

#include "ops/softMax_op.h"

namespace kuiper_infer {
    SoftMaxOperator::SoftMaxOperator() : RuntimeOperator(OpType::kOperatorSoftMax) {}

    SoftMaxOperator::SoftMaxOperator(int dim) : RuntimeOperator(OpType::kOperatorSoftMax), dim_(dim) {}

    void SoftMaxOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void SoftMaxOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> SoftMaxOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorSoftMax);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<SoftMaxOperator>();
        return runtimeOperator;
    }

    uint32_t SoftMaxOperator::getDim() const {
        return dim_;
    }

    void SoftMaxOperator::setDim(uint32_t dim) {
        dim_ = dim;
    }

    RuntimeOperatorRegistererWrapper kSoftMaxOperator(OpType::kOperatorSoftMax, SoftMaxOperator::CreateInstance);
}