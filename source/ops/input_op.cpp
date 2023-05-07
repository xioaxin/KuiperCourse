//
// Created by zpx on 2023/04/19.
//
#include "ops/input_op.h"

namespace kuiper_infer {
    InputOperator::InputOperator() : RuntimeOperator(OpType::kOperatorInput) {}

    void InputOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void InputOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> InputOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorInput);
        std::shared_ptr<RuntimeOperator> inputOperator = std::make_shared<InputOperator>();
        return inputOperator;
    }

    RuntimeOperatorRegistererWrapper kInputOperator(OpType::kOperatorInput, InputOperator::CreateInstance);
}