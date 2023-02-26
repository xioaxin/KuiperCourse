//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_HARDSIGMOID_OP_H
#define KUIPER_COURSE_HARDSIGMOID_OP_H

#include "ops.h"
#include <cstdint>

namespace kuiper_infer {
    class HardSigmoidOperator : public Operator {
    public:
        explicit HardSigmoidOperator();
        ~HardSigmoidOperator() override = default;
    };
}
#endif //KUIPER_COURSE_HARDSIGMOID_OP_H
