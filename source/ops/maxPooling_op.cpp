//
// Created by zpx on 2023/02/03.
//
#include <utility>

#include "ops/maxPooling_op.h"

namespace kuiper_infer {
    MaxPoolingOperator::MaxPoolingOperator() : RuntimeOperator(OpType::kOperatorMaxPooling) {}

    MaxPoolingOperator::MaxPoolingOperator(std::vector<int> &kernel_size, std::vector<int> &padding_size,
                                           std::vector<int> &stride,  std::vector<int> &dilation) :
            RuntimeOperator(OpType::kOperatorMaxPooling), kernel_size_(std::move(kernel_size)), padding_size_(padding_size),
            stride_(std::move(stride)), dilation_(dilation) {
    }

    void MaxPoolingOperator::initialParameter(const std::map<std::string,std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->kernel_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("kernel_size").get())->value;
        this->dilation_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("dilation").get())->value;
        this->padding_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("padding").get())->value;
        this->stride_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("stride").get())->value;
    }

    void MaxPoolingOperator::initialAttribute(
            const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> MaxPoolingOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorMaxPooling);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<MaxPoolingOperator>();
        return runtimeOperator;
    }

    const std::vector<int> &MaxPoolingOperator::getKernelSize() const {
        return kernel_size_;
    }

    void MaxPoolingOperator::setKernelSize(const std::vector<int> &kernelSize) {
        kernel_size_ = kernelSize;
    }

    const std::vector<int> &MaxPoolingOperator::getPaddingSize() const {
        return padding_size_;
    }

    void MaxPoolingOperator::setPaddingSize(const std::vector<int> &paddingSize) {
        padding_size_ = paddingSize;
    }

    const std::vector<int> &MaxPoolingOperator::getStride() const {
        return stride_;
    }

    void MaxPoolingOperator::setStride(const std::vector<int> &stride) {
        stride_ = stride;
    }

    const std::vector<int> &MaxPoolingOperator::getDilation() const {
        return dilation_;
    }

    void MaxPoolingOperator::setDilation(const std::vector<int> &dilation) {
        dilation_ = dilation;
    }

    RuntimeOperatorRegistererWrapper kMaxPoolingOperator(OpType::kOperatorMaxPooling,
                                                         MaxPoolingOperator::CreateInstance);
}