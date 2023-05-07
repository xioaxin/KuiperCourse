//
// Created by zpx on 2023/02/28.
//
#include "ops/silu_op.h"

namespace kuiper_infer {
    SiluOperator::SiluOperator() : RuntimeOperator(OpType::kOperatorSilu) {}

    void SiluOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void SiluOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> SiluOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorSilu);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<SiluOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kSiluOperator(OpType::kOperatorSilu, SiluOperator::CreateInstance);
}