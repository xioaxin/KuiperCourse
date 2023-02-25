//
// Created by zpx on 2023/02/25.
//
#include "ops/batchNorm_op.h"

namespace kuiper_infer {
    BatchNormOperator::BatchNormOperator(float eps) : Operator(OpType::kOperatorBatchNorm), eps_(eps) {
    }

    const std::vector<float> BatchNormOperator::getAffineAlpha() const {
        return affine_alpha_;
    }

    const std::vector<float> BatchNormOperator::getAffineBata() const {
        return affine_beta_;
    }

    const float BatchNormOperator::getEps() const {
        return eps_;
    }

    const sftensor BatchNormOperator::getMeanValue() const {
        return mean_value_;
    }

    const sftensor BatchNormOperator::getVarValue() const {
        return var_value_;
    }

    void BatchNormOperator::setAffineAlpha(const std::vector<float> &affine_alpha) {
        this->affine_alpha_ = affine_alpha;
    }

    void BatchNormOperator::setAffineBata(const std::vector<float> &affine_beta) {
        this->affine_beta_ = affine_beta;
    }

    void BatchNormOperator::setEps(float eps) {
        this->eps_ = eps;
    }

    void BatchNormOperator::setMeanValue(const sftensor &mean_value) {
        this->mean_value_ = mean_value;
    }

    void BatchNormOperator::setVarValue(const sftensor &var_value) {
        this->var_value_ = var_value;
    }
}