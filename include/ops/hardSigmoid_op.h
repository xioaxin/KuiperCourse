//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_HARDSIGMOID_OP_H
#define KUIPER_COURSE_HARDSIGMOID_OP_H

#include <cstdint>
#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class HardSigmoidOperator : public RuntimeOperator {
    public:
        HardSigmoidOperator();

        ~HardSigmoidOperator() {};
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    };
}
#endif //KUIPER_COURSE_HARDSIGMOID_OP_H
