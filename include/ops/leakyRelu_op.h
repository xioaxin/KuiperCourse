//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_LEAKYRELU_H
#define KUIPER_COURSE_LEAKYRELU_H

#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class LeakyReluOperator : public RuntimeOperator {
    public:
        LeakyReluOperator();

        ~LeakyReluOperator() {};
        explicit LeakyReluOperator(float thresh, float alpha = 0.01f);
        void set_thresh(const float thresh);
        float get_thresh() const;
        float getAlpha() const;
        void setAlpha(float alpha);
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        float thresh_ = 0.f;
        float alpha_ = 0.01f;
    };
}
#endif //KUIPER_COURSE_LEAKYRELU_H