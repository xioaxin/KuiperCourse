//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_CONVOLUTION_OP_H
#define KUIPER_COURSE_CONVOLUTION_OP_H

#include "ops.h"
#include <cstdint>
#include <vector>
#include "data/tensor.hpp"

namespace kuiper_infer {
    class ConvolutionOperator : public Operator {
    public:
        explicit ConvolutionOperator(bool use_bias, uint32_t groups, uint32_t stride_h,
                                     uint32_t stride_w, uint32_t padding_h, uint32_t padding_w);
        void setWeights(std::vector<std::shared_ptr<ftensor>> &weight);
        void setBias(std::vector<std::shared_ptr<ftensor>> &bias);
        const std::vector<std::shared_ptr<ftensor >> &getWeight() const;
        const std::vector<std::shared_ptr<ftensor >> &getBias() const;
        bool isUseBias() const;
        void setUseBias(bool use_bias);
        const uint32_t getGroups();
        void setGroups(uint32_t groups);
        const uint32_t getPadding_H() const;
        void setPadding_H(uint32_t padding_h);
        const uint32_t getPadding_W() const;
        void setPadding_W(uint32_t padding_w);
        const uint32_t getStride_H();
        void setStride_H(uint32_t stride_h);
        const uint32_t getStride_W();
        void setStride_W(uint32_t stride_w);
    private:
        bool use_bias_ = false;
        uint32_t groups_ = 1;
        uint32_t stride_h_ = 1;
        uint32_t stride_w_ = 1;
        uint32_t padding_h_ = 0;
        uint32_t padding_w_ = 0;
        std::vector<std::shared_ptr<ftensor>> weight_;
        std::vector<std::shared_ptr<ftensor>> bias_;
    };
}
#endif //KUIPER_COURSE_CONVOLUTION_OP_H
