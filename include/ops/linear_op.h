//
// Created by zpx on 2023/02/28.
//

#ifndef KUIPER_COURSE_LINEAR_OP_H
#define KUIPER_COURSE_LINEAR_OP_H

#include <vector>
#include "data/tensor.hpp"
#include <stdint.h>
#include <memory>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class LinearOperator : public RuntimeOperator {
    public:
        LinearOperator();
        explicit LinearOperator(uint32_t input_feature, uint32_t output_feature);

        virtual ~LinearOperator() {
        };
        void setInputFeature(uint32_t input_feature);
        uint32_t getInputFeature() const;
        void setOutputFeature(uint32_t output_feature);
        uint32_t getOutputFeature() const;
        bool isUseBias() const;
        void setUseBias(bool use_bias);
        bool getUseBias() const;
        std::vector<sftensor> getWeight() const;
        void setWeight(const std::vector<sftensor> &weight);
        std::vector<sftensor> getBias() const;
        void setBias(const std::vector<sftensor> &bias);
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(std::string type);
    private:
        std::vector<sftensor> weight_;
        std::vector<sftensor> bias_;
        bool use_bias_ = false;
        uint32_t input_feature_;
        uint32_t output_feature_;
    };
}
#endif //KUIPER_COURSE_LINEAR_OP_H
