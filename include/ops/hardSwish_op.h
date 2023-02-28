//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_HARDSWISH_OP_H
#define KUIPER_COURSE_HARDSWISH_OP_H

#include "ops.h"

namespace kuiper_infer {
    class HardSwishOperator : public Operator {
    public:
        explicit HardSwishOperator();
        ~HardSwishOperator() override = default;
    };
}
#endif //KUIPER_COURSE_HARDSWISH_OP_H
