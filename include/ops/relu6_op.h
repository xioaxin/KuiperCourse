//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_RELU6_H
#define KUIPER_COURSE_RELU6_H

#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class Relu6Operator : public RuntimeOperator {
    public:
        Relu6Operator();

        ~Relu6Operator() {};
        explicit Relu6Operator(float thresh);
        void set_thresh(const float thresh);
        float get_thresh() const;
        void initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        float thresh_ = 0.f;
    };
}
#endif //KUIPER_COURSE_RELU6_H
