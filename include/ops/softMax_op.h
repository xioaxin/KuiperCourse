//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_SOFTMAX_OP_H
#define KUIPER_COURSE_SOFTMAX_OP_H

#include "ops.h"
#include <cstdint>

namespace kuiper_infer {
    class SoftMaxOperator : public Operator {
    public:
        ~SoftMaxOperator() override = default;
        SoftMaxOperator();
    };
}
#endif //KUIPER_COURSE_SOFTMAX_OP_H
