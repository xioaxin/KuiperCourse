//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_SOFTMAX_OP_H
#define KUIPER_COURSE_SOFTMAX_OP_H

#include <cstdint>
#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class SoftMaxOperator : public RuntimeOperator {
    public:
        ~SoftMaxOperator() {};
        SoftMaxOperator();
        explicit SoftMaxOperator(int dim);
        void initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        uint32_t dim_ = 0;
    public:
        uint32_t getDim() const;
        void setDim(uint32_t dim);
    };
}
#endif //KUIPER_COURSE_SOFTMAX_OP_H
