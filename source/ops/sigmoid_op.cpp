//
// Created by zpx on 2023/01/01.
//
#include "ops/sigmoid_op.h"

namespace kuiper_infer {
    SigmoidOperator::SigmoidOperator() : RuntimeOperator(OpType::kOperatorSigmoid) {}

    void SigmoidOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void SigmoidOperator::initialAttribute(const std::map<std::string,std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> SigmoidOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorSigmoid);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<SigmoidOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kSigmoidOperator(OpType::kOperatorSigmoid, SigmoidOperator::CreateInstance);
}