//
// Created by zpx on 2023/03/01.
//
#include "ops/upSample_op.h"

namespace kuiper_infer {
    UpSampleOperator::UpSampleOperator(const float scale_h, const float scale_w,
                                       const kuiper_infer::UpSampleMode upSampleMode) :
            Operator(OpType::kOperatorUpSample), scale_h_(scale_h), scale_w_(scale_w), upSampleMode1_(upSampleMode) {};

    void UpSampleOperator::setScaleH(const float scale_h) {
        this->scale_h_ = scale_h;
    }

    void UpSampleOperator::setScaleW(const float scale_w) {
        this->scale_w_ = scale_w;
    }

    const float UpSampleOperator::getScaleH() const {
        return scale_h_;
    }

    const float UpSampleOperator::getScaleW() const {
        return scale_w_;
    }

    void UpSampleOperator::setUpSampleModel(const kuiper_infer::UpSampleMode upSampleMode) {
        this->upSampleMode1_ = upSampleMode;
    }

    const UpSampleMode UpSampleOperator::getUpSampleModel() const {
        return this->upSampleMode1_;
    }
}