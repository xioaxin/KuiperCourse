//
// Created by zpx on 2023/02/21.
//
#include <utility>

#include "ops/adaptiveAvgPooling_op.h"

namespace kuiper_infer {
    AdaptiveAvgPoolingOperator::AdaptiveAvgPoolingOperator() : RuntimeOperator(OpType::kOperatorAdaptiveAvgPooling) {}

    AdaptiveAvgPoolingOperator::AdaptiveAvgPoolingOperator(std::vector<int> output_size) : RuntimeOperator(
            OpType::kOperatorAdaptiveAvgPooling), output_size_(std::move(output_size)) {}

    void AdaptiveAvgPoolingOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->output_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("output_size"))->value;
    }

    void AdaptiveAvgPoolingOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> AdaptiveAvgPoolingOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorAdaptiveAvgPooling);
        std::shared_ptr<RuntimeOperator> adaptiveAvgPoolingOperator = std::make_shared<AdaptiveAvgPoolingOperator>();
        return adaptiveAvgPoolingOperator;
    }

    const std::vector<int> &AdaptiveAvgPoolingOperator::getOutputSize() const {
        return output_size_;
    }

    void AdaptiveAvgPoolingOperator::setOutputSize(const std::vector<int> &outputSize) {
        output_size_ = outputSize;
    }

    RuntimeOperatorRegistererWrapper kAdaptiveAvgPoolingOperator(OpType::kOperatorAdaptiveAvgPooling, AdaptiveAvgPoolingOperator::CreateInstance);
}