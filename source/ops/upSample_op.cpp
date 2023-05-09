//
// Created by zpx on 2023/03/01.
//
#include "ops/upSample_op.h"

namespace kuiper_infer {
    UpSampleOperator::UpSampleOperator() : RuntimeOperator(OpType::kOperatorUpSample) {}

    UpSampleOperator::UpSampleOperator(const float scale_h, const float scale_w,
                                       const kuiper_infer::UpSampleMode upSampleMode) :
            RuntimeOperator(OpType::kOperatorUpSample), scale_h_(scale_h), scale_w_(scale_w),
            upSampleMode1_(upSampleMode) {};

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

    void UpSampleOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
    }

    void UpSampleOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }
    std::shared_ptr<RuntimeOperator> UpSampleOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorUpSample);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<UpSampleOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kUpSampleOperator(OpType::kOperatorUpSample, UpSampleOperator::CreateInstance);
}