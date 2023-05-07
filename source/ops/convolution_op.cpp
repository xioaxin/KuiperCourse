//
// Created by zpx on 2023/02/21.
//
#include "ops/convolution_op.h"

namespace kuiper_infer {
    ConvolutionOperator::ConvolutionOperator() : RuntimeOperator(OpType::kOperatorConvolution) {}

    ConvolutionOperator::ConvolutionOperator(
            uint32_t in_channels, uint32_t out_channels, const std::vector<int> &kernel_size, uint32_t groups,
            const std::vector<int> &padding, const std::vector<int> &stride, const std::vector<int> &dilation,
            bool use_bias, std::string padding_mode) :
            RuntimeOperator(OpType::kOperatorConvolution),
            use_bias_(use_bias), groups_(groups),
            stride_(stride), padding_(padding), dilation_(dilation),
            in_channels_(in_channels),
            out_channels_(out_channels), padding_mode_(padding_mode) {}

    const uint32_t ConvolutionOperator::getGroups() {
        return groups_;
    }

    void ConvolutionOperator::setGroups(uint32_t groups) {
        this->groups_ = groups;
    }

    const std::vector<std::shared_ptr<ftensor>> &ConvolutionOperator::getWeight() const {
        return this->weight_;
    }

    void ConvolutionOperator::setWeights(std::vector<std::shared_ptr<ftensor>> weight) {
        this->weight_ = weight;
    }

    const std::vector<std::shared_ptr<ftensor>> &ConvolutionOperator::getBias() const {
        return this->bias_;
    }

    void ConvolutionOperator::setBias(std::vector<std::shared_ptr<ftensor>> bias) {
        this->bias_ = bias;
    }

    bool ConvolutionOperator::isUseBias() const {
        return use_bias_;
    }

    void ConvolutionOperator::setUseBias(bool use_bias) {
        this->use_bias_ = use_bias;
    }

    void ConvolutionOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->use_bias_ = dynamic_cast<RuntimeParameterBool *>(runtimeParameter.at("bias"))->value;
        this->dilation_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("dilation"))->value;
        this->groups_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("groups"))->value;
        this->in_channels_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("in_channels"))->value;
        this->out_channels_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("out_channels"))->value;
        this->padding_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("padding"))->value;
        this->padding_mode_ = dynamic_cast<RuntimeParameterString *>(runtimeParameter.at("padding_mode"))->value;
        this->kernel_size_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("kernel_size"))->value;
        this->stride_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("stride"))->value;
    }

    void ConvolutionOperator::initialAttribute(
            const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
        setWeights(RuntimeAttribute::get_value(runtimeAttribute.at("weight")));
        if (use_bias_) setBias(RuntimeAttribute::get_value(runtimeAttribute.at("bias")));
    }

    std::shared_ptr<RuntimeOperator> ConvolutionOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorConvolution);
        std::shared_ptr<RuntimeOperator> convOperator = std::make_shared<ConvolutionOperator>();
        return convOperator;
    }

    const std::vector<int> &ConvolutionOperator::getStride() const {
        return stride_;
    }

    void ConvolutionOperator::setStride(const std::vector<int> &stride) {
        stride_ = stride;
    }

    const std::vector<int> &ConvolutionOperator::getPadding() const {
        return padding_;
    }

    void ConvolutionOperator::setPadding(const std::vector<int> &padding) {
        padding_ = padding;
    }

    const std::vector<int> &ConvolutionOperator::getDilation() const {
        return dilation_;
    }

    void ConvolutionOperator::setDilation(const std::vector<int> &dilation) {
        dilation_ = dilation;
    }

    const std::vector<int> &ConvolutionOperator::getKernelSize() const {
        return kernel_size_;
    }

    void ConvolutionOperator::setKernelSize(const std::vector<int> &kernelSize) {
        kernel_size_ = kernelSize;
    }

    uint32_t ConvolutionOperator::getInChannels() const {
        return in_channels_;
    }

    void ConvolutionOperator::setInChannels(uint32_t inChannels) {
        in_channels_ = inChannels;
    }

    uint32_t ConvolutionOperator::getOutChannels() const {
        return out_channels_;
    }

    void ConvolutionOperator::setOutChannels(uint32_t outChannels) {
        out_channels_ = outChannels;
    }

    const std::string &ConvolutionOperator::getPaddingMode() const {
        return padding_mode_;
    }

    void ConvolutionOperator::setPaddingMode(const std::string &paddingMode) {
        padding_mode_ = paddingMode;
    }

    RuntimeOperatorRegistererWrapper kConvolutionOperator(OpType::kOperatorConvolution,ConvolutionOperator::CreateInstance);
}