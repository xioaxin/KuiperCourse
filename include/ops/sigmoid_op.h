//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_SIGMOID_OP_H
#define KUIPER_COURSE_SIGMOID_OP_H

#include "ops.h"

namespace kuiper_infer {
    class SigmoidOperator : public Operator {
    public:
        ~SigmoidOperator() override = default;
        explicit SigmoidOperator();
    };
}
#endif //KUIPER_COURSE_SIGMOID_OP_H
