//
// Created by zpx on 2023/04/19.
//

#ifndef KUIPER_COURSE_INPUT_OP_H
#define KUIPER_COURSE_INPUT_OP_H

#include "factory/operator_factory.h"

namespace kuiper_infer {
    class InputOperator : public RuntimeOperator {
    public:
        InputOperator();

        ~InputOperator() {};
        void initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    };
}
#endif //KUIPER_COURSE_INPUT_OP_H
