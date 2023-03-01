//
// Created by zpx on 2023/02/28.
//
#include "ops/linear_op.h"

namespace kuiper_infer {
    LinearOperator::LinearOperator(const uint32_t input_feature, uint32_t output_feature) : Operator(
            OpType::kOperatorLinear), input_feature_(input_feature), output_feature_(output_feature) {};

    const uint32_t LinearOperator::getInputFeature() const {
        return input_feature_;
    }

    const uint32_t LinearOperator::getOutputFeature() const {
        return output_feature_;
    }

    void LinearOperator::setInputFeature(const uint32_t input_feature) {
        this->input_feature_ = input_feature;
    }

    void LinearOperator::setOutputFeature(const uint32_t output_feature) {
        this->output_feature_ = output_feature;
    }

    void LinearOperator::setWeights(const std::shared_ptr<ftensor> &weight) {
        this->weight_ = weight;
    }

    const std::shared_ptr<ftensor> LinearOperator::getWeights() const {
        return this->weight_;
    }

    void LinearOperator::setBias(const std::shared_ptr<ftensor> &bias) {
        this->bias_ = bias;
    }

    const std::shared_ptr<ftensor> LinearOperator::getBias() const {
        return this->bias_;
    }

    const bool LinearOperator::isUseBias() const {
        return this->use_bias_;
    }

    void LinearOperator::setUseBias(const bool use_bias) {
        this->use_bias_ = use_bias;
    }

    const bool LinearOperator::getUseBias() const {
        return this->use_bias_;
    }
}