//
// Created by zpx on 2023/02/21.
//
#include "ops/convolution_op.h"

namespace kuiper_infer {
    ConvolutionOperator::ConvolutionOperator(bool use_bias, uint32_t groups, uint32_t stride_h, uint32_t stride_w,
                                             uint32_t padding_h, uint32_t padding_w) :
            Operator(OpType::kOperatorConvolution), use_bias_(use_bias), groups_(groups), stride_h_(stride_h),
            stride_w_(stride_w), padding_h_(padding_h), padding_w_(padding_w) {
    }

    const uint32_t ConvolutionOperator::getGroups() {
        return groups_;
    }

    void ConvolutionOperator::setGroups(uint32_t groups) {
        this->groups_ = groups;
    }

    const uint32_t ConvolutionOperator::getPadding_H() const {
        return padding_h_;
    }

    void ConvolutionOperator::setPadding_H(uint32_t padding_h) {
        this->padding_h_ = padding_h;
    }

    const uint32_t ConvolutionOperator::getPadding_W() const {
        return padding_w_;
    }

    void ConvolutionOperator::setPadding_W(uint32_t padding_w) {
        this->padding_w_ = padding_w;
    }

    const uint32_t ConvolutionOperator::getStride_H() {
        return stride_h_;
    }

    void ConvolutionOperator::setStride_H(uint32_t stride_h) {
        this->stride_h_ = stride_h;
    }

    const uint32_t ConvolutionOperator::getStride_W() {
        return stride_w_;
    }

    void ConvolutionOperator::setStride_W(uint32_t stride_w) {
        this->stride_w_ = stride_w;
    }

    const std::vector<std::shared_ptr<ftensor>> &ConvolutionOperator::getWeight() const {
        return this->weight_;
    }

    void ConvolutionOperator::setWeights(std::vector<std::shared_ptr<ftensor>> &weight) {
        this->weight_ = weight;
    }

    const std::vector<std::shared_ptr<ftensor>> &ConvolutionOperator::getBias() const {
        return this->bias_;
    }

    void ConvolutionOperator::setBias(std::vector<std::shared_ptr<ftensor>> &bias) {
        this->bias_ = bias;
    }

    bool ConvolutionOperator::isUseBias() const {
        return use_bias_;
    }

    void ConvolutionOperator::setUseBias(bool use_bias) {
        this->use_bias_ = use_bias;
    }
}