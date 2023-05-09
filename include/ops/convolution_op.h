//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_CONVOLUTION_OP_H
#define KUIPER_COURSE_CONVOLUTION_OP_H

#include <cstdint>
#include <vector>
#include "data/tensor.hpp"
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class ConvolutionOperator : public RuntimeOperator {
    public:
        ConvolutionOperator();
        explicit ConvolutionOperator(uint32_t in_channels, uint32_t out_channels, const std::vector<int> &kernel_size = {3, 3},
                                     uint32_t groups = 1, const std::vector<int> &padding = {0, 0}, const std::vector<int> &stride = {1, 1},
                                     const std::vector<int> &dilation = {0, 0}, bool use_bias = false, std::string padding_mode = "same");
        ~ConvolutionOperator() override = default;
        void setWeights(std::vector<std::shared_ptr<ftensor>> weight);
        void setBias(std::vector<std::shared_ptr<ftensor>> bias);
        [[nodiscard]] const std::vector<std::shared_ptr<ftensor >> &getWeight() const;
        [[nodiscard]] const std::vector<std::shared_ptr<ftensor >> &getBias() const;
        bool isUseBias() const;
        void setUseBias(bool use_bias);
        const uint32_t getGroups();
        void setGroups(uint32_t groups);
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        bool use_bias_ = false;
        uint32_t groups_ = 1;
        std::vector<int> stride_;
        std::vector<int> padding_;
        std::vector<int> dilation_;
        std::vector<int> kernel_size_;
        uint32_t in_channels_ = 0;
        uint32_t out_channels_ = 0;
        std::string padding_mode_;
    public:
        [[nodiscard]] const std::vector<int> &getStride() const;
        void setStride(const std::vector<int> &stride);
        [[nodiscard]] const std::vector<int> &getPadding() const;
        void setPadding(const std::vector<int> &padding);
        [[nodiscard]] const std::vector<int> &getDilation() const;
        void setDilation(const std::vector<int> &dilation);
        [[nodiscard]] const std::vector<int> &getKernelSize() const;
        void setKernelSize(const std::vector<int> &kernelSize);
        [[nodiscard]] uint32_t getInChannels() const;
        void setInChannels(uint32_t inChannels);
        [[nodiscard]] uint32_t getOutChannels() const;
        void setOutChannels(uint32_t outChannels);
        [[nodiscard]] const std::string &getPaddingMode() const;
        void setPaddingMode(const std::string &paddingMode);
    private:
        std::vector<std::shared_ptr<ftensor>> weight_;
        std::vector<std::shared_ptr<ftensor>> bias_;
    };
}
#endif //KUIPER_COURSE_CONVOLUTION_OP_H
