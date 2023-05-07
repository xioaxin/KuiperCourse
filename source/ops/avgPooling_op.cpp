//
// Created by zpx on 2023/02/03.
//
#include "ops/avgPooling_op.h"

namespace kuiper_infer {
    AvgPoolingOperator::AvgPoolingOperator() : RuntimeOperator(OpType::kOperatorAvgPooling) {}

    AvgPoolingOperator::AvgPoolingOperator(std::vector<int> &kernel_size, std::vector<int> padding_size, std::vector<int> &stride,
                                           std::vector<int> &dilation) : RuntimeOperator(OpType::kOperatorAvgPooling), kernel_size_
            (kernel_size), padding_size_(padding_size), stride_(stride), dilation_(dilation) {
    }

    void AvgPoolingOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->kernel_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("kernel_size"))->value;
        this->dilation_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("dilation"))->value;
        this->padding_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("padding"))->value;
        this->stride_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("stride"))->value;
    }

    void AvgPoolingOperator::initialAttribute(
            const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {}

    std::shared_ptr<RuntimeOperator> AvgPoolingOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorAvgPooling);
        std::shared_ptr<RuntimeOperator> avgPoolingOperator = std::make_shared<AvgPoolingOperator>();
        return avgPoolingOperator;
    }

    const std::vector<int> &AvgPoolingOperator::getKernelSize() const {
        return kernel_size_;
    }

    void AvgPoolingOperator::setKernelSize(const std::vector<int> &kernelSize) {
        kernel_size_ = kernelSize;
    }

    const std::vector<int> &AvgPoolingOperator::getPaddingSize() const {
        return padding_size_;
    }

    void AvgPoolingOperator::setPaddingSize(const std::vector<int> &paddingSize) {
        padding_size_ = paddingSize;
    }

    const std::vector<int> &AvgPoolingOperator::getStride() const {
        return stride_;
    }

    void AvgPoolingOperator::setStride(const std::vector<int> &stride) {
        stride_ = stride;
    }

    const std::vector<int> &AvgPoolingOperator::getDilation() const {
        return dilation_;
    }

    void AvgPoolingOperator::setDilation(const std::vector<int> &dilation) {
        dilation_ = dilation;
    }

    RuntimeOperatorRegistererWrapper kAvgPoolingOperator(OpType::kOperatorAvgPooling,
                                                         AvgPoolingOperator::CreateInstance);
}