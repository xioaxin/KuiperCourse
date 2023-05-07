//
// Created by zpx on 2023/02/28.
//
#include "ops/linear_op.h"

namespace kuiper_infer {
    LinearOperator::LinearOperator() : RuntimeOperator(OpType::kOperatorLinear) {}

    LinearOperator::LinearOperator(const uint32_t input_feature, uint32_t output_feature) : RuntimeOperator(
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

    const bool LinearOperator::isUseBias() const {
        return this->use_bias_;
    }

    void LinearOperator::setUseBias(const bool use_bias) {
        this->use_bias_ = use_bias;
    }

    const bool LinearOperator::getUseBias() const {
        return this->use_bias_;
    }

    void LinearOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->use_bias_ = dynamic_cast<RuntimeParameterBool *>(runtimeParameter.at("bias"))->value;
        this->input_feature_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("in_features"))->value;
        this->output_feature_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("out_features"))->value;
    }

    void
    LinearOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
        if (runtimeAttribute.find("weight") != runtimeAttribute.end())
            setWeight(RuntimeAttribute::get_value_matrix(runtimeAttribute.at("weight"), this->input_operands_seq[0]->shapes[0]));
        if (use_bias_ && runtimeAttribute.find("bias") != runtimeAttribute.end())
            setBias(RuntimeAttribute::get_value_vector(runtimeAttribute.at("bias"),this->input_operands_seq[0]->shapes[0]));
    }

    std::shared_ptr<RuntimeOperator> LinearOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorLinear);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<LinearOperator>();
        return runtimeOperator;
    }

    const std::vector<sftensor> &LinearOperator::getWeight() const {
        return weight_;
    }

    void LinearOperator::setWeight(const std::vector<sftensor> &weight) {
        weight_ = std::move(weight);
    }

    const std::vector<sftensor> &LinearOperator::getBias() const {
        return bias_;
    }

    void LinearOperator::setBias(const std::vector<sftensor> &bias) {
        bias_ = std::move(bias);
    }

    RuntimeOperatorRegistererWrapper kLinearOperator(OpType::kOperatorLinear, LinearOperator::CreateInstance);
}