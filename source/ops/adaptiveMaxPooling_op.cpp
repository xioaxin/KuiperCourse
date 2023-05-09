//
// Created by zpx on 2023/02/21.
//
#include <utility>
#include "ops/adaptiveMaxPooling_op.h"

namespace kuiper_infer {
    AdaptiveMaxPoolingOperator::AdaptiveMaxPoolingOperator() : RuntimeOperator(OpType::kOperatorAdaptiveMaxPooling) {}

    AdaptiveMaxPoolingOperator::AdaptiveMaxPoolingOperator(std::vector<int> output_size) : RuntimeOperator(
            OpType::kOperatorAdaptiveMaxPooling), output_size_(std::move(output_size)) {}

    void
    AdaptiveMaxPoolingOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->output_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("output_size").get())->value;
    }

    void AdaptiveMaxPoolingOperator::initialAttribute(
            const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> AdaptiveMaxPoolingOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorAdaptiveMaxPooling);
        std::shared_ptr<RuntimeOperator> adaptiveMaxPoolingOperator = std::make_shared<AdaptiveMaxPoolingOperator>();
        return adaptiveMaxPoolingOperator;
    }

    const std::vector<int> &AdaptiveMaxPoolingOperator::getOutputSize() const {
        return output_size_;
    }

    void AdaptiveMaxPoolingOperator::setOutputSize(const std::vector<int> &outputSize) {
        output_size_ = outputSize;
    }

    RuntimeOperatorRegistererWrapper kAdaptiveMaxPoolingOperator(OpType::kOperatorAdaptiveMaxPooling,
                                                                 AdaptiveMaxPoolingOperator::CreateInstance);
}