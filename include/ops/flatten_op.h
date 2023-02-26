//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_FLATTEN_H
#define KUIPER_COURSE_FLATTEN_H

#include "ops.h"
#include <cstdint>

namespace kuiper_infer {
    class FlattenOperator : public Operator {
    public:
        explicit FlattenOperator(uint32_t start_dim, uint32_t end_dim);
        ~FlattenOperator() override = default;
        void setStartDim(const uint32_t start_dim);
        void setEndDim(const uint32_t end_dim);
        const uint32_t getStartDim() const;
        const uint32_t getEndDim() const;
    private:
        uint32_t start_dim_;
        uint32_t end_dim_;
    };
}
#endif //KUIPER_COURSE_FLATTEN_H
