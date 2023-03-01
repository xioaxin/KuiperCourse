//
// Created by zpx on 2023/02/28.
//

#ifndef KUIPER_COURSE_LINEAR_OP_H
#define KUIPER_COURSE_LINEAR_OP_H

#include "ops.h"
#include <vector>
#include "data/tensor.hpp"
#include <stdint.h>
#include <memory>

namespace kuiper_infer {
    class LinearOperator : public Operator {
    public:
        explicit LinearOperator(const uint32_t input_feature, uint32_t output_feature);
        ~LinearOperator() override = default;
        void setInputFeature(const uint32_t input_feature);
        const uint32_t getInputFeature() const;
        void setOutputFeature(const uint32_t output_feature);
        const uint32_t getOutputFeature() const;
        const bool isUseBias() const;
        void setUseBias(const bool use_bias);
        const bool getUseBias() const;
        void setWeights(const std::shared_ptr<ftensor> &weight);
        const std::shared_ptr<ftensor> getWeights() const;
        void setBias(const std::shared_ptr<ftensor> &bias);
        const std::shared_ptr<ftensor> getBias() const;
    private:
        std::shared_ptr<ftensor> weight_;
        std::shared_ptr<ftensor> bias_;
        bool use_bias_ = false;
        uint32_t input_feature_;
        uint32_t output_feature_;
    };
}
#endif //KUIPER_COURSE_LINEAR_OP_H
