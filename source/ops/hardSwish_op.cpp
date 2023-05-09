//
// Created by zpx on 2023/02/26.
//
#include "ops/hardSwish_op.h"

namespace kuiper_infer {
    HardSwishOperator::HardSwishOperator() : RuntimeOperator(OpType::kOperatorHardSwish) {}

    void HardSwishOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
    }

    void HardSwishOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> HardSwishOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorHardSwish);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<HardSwishOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kHardSwishOperator(OpType::kOperatorHardSwish, HardSwishOperator::CreateInstance);
}