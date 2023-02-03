//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_LEAKYRELU_H
#define KUIPER_COURSE_LEAKYRELU_H

#include "ops/ops.h"

namespace kuiper_infer {
    class LeakyReluOperator : public Operator {
    public:
        ~LeakyReluOperator() override = default;
        explicit LeakyReluOperator(float thresh);
        void set_thresh(const float thresh);
        float get_thresh() const;
    private:
        float thresh_ = 0.f;
    };
}
#endif //KUIPER_COURSE_LEAKYRELU_H
