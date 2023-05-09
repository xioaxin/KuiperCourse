//
// Created by zpx on 2023/02/18.
//

#ifndef KUIPER_COURSE_AVGPOOLING_OP_H
#define KUIPER_COURSE_AVGPOOLING_OP_H

#include <cstdint>
#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class AvgPoolingOperator : public RuntimeOperator {
    public:
        AvgPoolingOperator();

        ~AvgPoolingOperator() {};
        explicit AvgPoolingOperator(std::vector<int> &kernel_size, std::vector<int> padding_size,
                                    std::vector<int> &stride, std::vector<int> &dilation);
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(std::string type);
    private:
        std::vector<int> kernel_size_;
        std::vector<int> padding_size_;
        std::vector<int> stride_;
    public:
        const std::vector<int> &getKernelSize() const;
        void setKernelSize(const std::vector<int> &kernelSize);
        const std::vector<int> &getPaddingSize() const;
        void setPaddingSize(const std::vector<int> &paddingSize);
        const std::vector<int> &getStride() const;
        void setStride(const std::vector<int> &stride);
        const std::vector<int> &getDilation() const;
        void setDilation(const std::vector<int> &dilation);
    private:
        std::vector<int> dilation_;
    };
}
#endif //KUIPER_COURSE_AVGPOOLING_OP_H
