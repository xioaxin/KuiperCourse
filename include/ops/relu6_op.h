//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_RELU6_H
#define KUIPER_COURSE_RELU6_H

#include "ops/ops.h"

namespace kuiper_infer {
    class Relu6Operator : public Operator {
    public:
        ~Relu6Operator() override = default;
        explicit Relu6Operator(float thresh);
        void set_thresh(const float thresh);
        float get_thresh() const;
    private:
        float thresh_ = 0.f;
    };
}
#endif //KUIPER_COURSE_RELU6_H
