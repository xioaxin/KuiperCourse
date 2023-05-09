//
// Created by zpx on 2023/02/26.
//
#include "ops/hardSigmoid_op.h"

namespace kuiper_infer {
    HardSigmoidOperator::HardSigmoidOperator() : RuntimeOperator(OpType::kOperatorHardSigmoid) {}

    void HardSigmoidOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
    }

    void HardSigmoidOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> HardSigmoidOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorSigmoid);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<HardSigmoidOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kHardSigmoidOperator(OpType::kOperatorHardSigmoid, HardSigmoidOperator::CreateInstance);
}