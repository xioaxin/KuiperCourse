//
// Created by zpx on 2022/12/26.
//
#include "ops/relu_op.h"

namespace kuiper_infer {
    ReluOperator::ReluOperator() : RuntimeOperator(OpType::kOperatorRelu) {}

    ReluOperator::ReluOperator(float thresh) : RuntimeOperator(OpType::kOperatorRelu), thresh_(thresh) {};

    void ReluOperator::set_thresh(float thresh) {
        thresh_ = thresh;
    }

    float ReluOperator::get_thresh() const { return thresh_; }

    void ReluOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {

    }

    void ReluOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {

    }

    std::shared_ptr<RuntimeOperator> ReluOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorRelu);
        std::shared_ptr<RuntimeOperator> reluOperator = std::make_shared<ReluOperator>();
        return reluOperator;
    }

    RuntimeOperatorRegistererWrapper kReluOperator(OpType::kOperatorRelu, ReluOperator::CreateInstance);
}