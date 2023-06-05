//
// Created by zpx on 2023/01/01.
//
#include "ops/leakyRelu_op.h"

namespace kuiper_infer {
    LeakyReluOperator::LeakyReluOperator() : RuntimeOperator(OpType::kOperatorLeakyRelu) {}

    LeakyReluOperator::LeakyReluOperator(float thresh, float alpha) : RuntimeOperator(OpType::kOperatorLeakyRelu),
                                                                      thresh_(thresh), alpha_(alpha) {};

    void LeakyReluOperator::set_thresh(const float thresh) { thresh_ = thresh; }

    float LeakyReluOperator::get_thresh() const { return thresh_; }

    float LeakyReluOperator::getAlpha() const { return alpha_; }

    void LeakyReluOperator::setAlpha(float alpha) { this->alpha_ = alpha; }

    void LeakyReluOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {}

    void LeakyReluOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> LeakyReluOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorLeakyRelu);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<LeakyReluOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kLeakyReluOperator(OpType::kOperatorLeakyRelu, LeakyReluOperator::CreateInstance);
}