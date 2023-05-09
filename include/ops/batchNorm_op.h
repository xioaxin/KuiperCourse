//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_BATCHNORM_OP_H
#define KUIPER_COURSE_BATCHNORM_OP_H

#include <vector>
#include "data/load_data.hpp"
#include "data/tensor.hpp"
#include <cstdint>
#include "factory/operator_factory.h"
namespace kuiper_infer {
    class BatchNormOperator : public RuntimeOperator {
    public:
        BatchNormOperator();

        ~BatchNormOperator() {};
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
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        float eps_ = 1e-5;
        sftensor mean_value_;
        sftensor var_value_;
        std::vector<float> affine_alpha_;
        std::vector<float> affine_beta_;
    };
}
#endif //KUIPER_COURSE_BATCHNORM_OP_H
