//
// Created by zpx on 2023/01/01.
//
#include "ops/relu6_op.h"

namespace kuiper_infer {
    Relu6Operator::Relu6Operator() : RuntimeOperator(OpType::kOperatorRelu6) {}

    Relu6Operator::Relu6Operator(float thresh) : RuntimeOperator(OpType::kOperatorRelu6), thresh_(thresh) {};

    void Relu6Operator::set_thresh(float thresh) {
        thresh_ = thresh;
    }

    float Relu6Operator::get_thresh() const { return thresh_; }

    void Relu6Operator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void Relu6Operator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> Relu6Operator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorRelu6);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<Relu6Operator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kRelu6Operator(OpType::kOperatorRelu6, Relu6Operator::CreateInstance);
}