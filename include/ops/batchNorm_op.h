//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_BATCHNORM_OP_H
#define KUIPER_COURSE_BATCHNORM_OP_H

#include <vector>
#include "data/load_data.hpp"
#include "data/tensor.hpp"
#include "ops.h"
#include <cstdint>

namespace kuiper_infer {
    class BatchNormOperator : public Operator {
    public:
        ~BatchNormOperator() override = default;
        explicit BatchNormOperator(float eps);
        void setMeanValue(const sftensor &mean_value);
        void setVarValue(const sftensor &var_value);
        void setAffineAlpha(const std::vector<float> &affine_alpha);
        void setAffineBata(const std::vector<float> &affine_beta);
        void setEps(float eps);
        const float getEps() const;
        const sftensor getMeanValue() const;
        const sftensor getVarValue() const;
        const std::vector<float> getAffineAlpha() const;
        const std::vector<float> getAffineBata() const;
    private:
        float eps_ = 1e-5;
        sftensor mean_value_;
        sftensor var_value_;
        std::vector<float> affine_alpha_;
        std::vector<float> affine_beta_;
    };
}
#endif //KUIPER_COURSE_BATCHNORM_OP_H
