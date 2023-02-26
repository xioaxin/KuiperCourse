//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_CAT_OP_H
#define KUIPER_COURSE_CAT_OP_H

#include "ops.h"
#include <cstdint>

namespace kuiper_infer {
    class CatOperator : public Operator {
    public:
        ~CatOperator() override = default;
        explicit CatOperator(uint32_t dim);
        void setDim(const uint32_t dim);
        const uint32_t getDim() const;
    private:
        uint32_t dim_;
    };
}
#endif //KUIPER_COURSE_CAT_OP_H
