//
// Created by zpx on 2022/12/26.
//

#ifndef KUIPER_COURSE_RELU_OP_H
#define KUIPER_COURSE_RELU_OP_H

#include "ops.h"

namespace kuiper_infer {
    class ReluOperator : public Operator {
    public:
        ~ReluOperator() override = default;
        explicit ReluOperator(float thresh);
        void set_thresh(float thresh);
        float get_thresh() const;
    private:
        float thresh_ = 0.f;
    };
}
#endif //KUIPER_COURSE_RELU_OP_H
