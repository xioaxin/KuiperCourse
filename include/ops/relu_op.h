//
// Created by zpx on 2022/12/26.
//

#ifndef KUIPER_COURSE_RELU_OP_H
#define KUIPER_COURSE_RELU_OP_H

#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class ReluOperator : public RuntimeOperator {
    public:
        ReluOperator();

        ~ReluOperator() {};
        explicit ReluOperator(float thresh);
        void set_thresh(float thresh);
        float get_thresh() const;
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        float thresh_ = 0.f;
    };
}
#endif //KUIPER_COURSE_RELU_OP_H
