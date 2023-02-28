//
// Created by zpx on 2023/02/28.
//

#ifndef KUIPER_COURSE_SILU_OP_H
#define KUIPER_COURSE_SILU_OP_H

#include "ops.h"

namespace kuiper_infer {
    class SiluOperator : public Operator {
    public:
        ~SiluOperator() override = default;
        explicit SiluOperator();
    };
}
#endif //KUIPER_COURSE_SILU_OP_H
