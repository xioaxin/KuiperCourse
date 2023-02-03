//
// Created by zpx on 2023/02/03.
//

#ifndef KUIPER_COURSE_MAXPOOLING_OP_H
#define KUIPER_COURSE_MAXPOOLING_OP_H

#include <cstdint>
#include "ops/ops.h"

namespace kuiper_infer {
    class MaxPoolingOperator : public Operator {
    public:
        ~MaxPoolingOperator() override = default;
        explicit MaxPoolingOperator(uint32_t pooling_h, uint32_t pooling_w, uint32_t stride_h,
                                    uint32_t stride_w, uint32_t padding_h, uint32_t padding_w);
        void set_pooling_h(uint32_t pool_h);
        void set_pooling_w(uint32_t pool_w);
        void set_stride_h(uint32_t stride_h);
        void set_stride_w(uint32_t stride_w);
        void set_padding_h(uint32_t padding_h);
        void set_padding_w(uint32_t padding_w);
        uint32_t get_pooling_h();
        uint32_t get_pooling_w();
        uint32_t get_stride_h();
        uint32_t get_stride_w();
        uint32_t get_padding_h();
        uint32_t get_padding_w();
    private:
        uint32_t pooling_h_;
        uint32_t pooling_w_;
        uint32_t stride_h_;
        uint32_t stride_w_;
        uint32_t padding_h_;
        uint32_t padding_w_;
    };
}
#endif //KUIPER_COURSE_MAXPOOLING_OP_H
