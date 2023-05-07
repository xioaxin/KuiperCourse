//
// Created by zpx on 2023/04/19.
//
#include "ops/output_op.h"

namespace kuiper_infer {
    OutputOperator::OutputOperator() : RuntimeOperator(OpType::kOperatorOutput) {}

    void OutputOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void OutputOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> OutputOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorOutput);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<OutputOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kOutputOperator(OpType::kOperatorOutput, OutputOperator::CreateInstance);
}