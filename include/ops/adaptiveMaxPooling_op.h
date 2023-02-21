//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_ADAPTIVEAVGPOOLING_H
#define KUIPER_COURSE_ADAPTIVEAVGPOOLING_H

#include "layer/layer.h"
#include "ops.h"
#include <cstdint>
#include "ops/ops.h"

namespace kuiper_infer {
    class AdaptiveMaxPoolingOperator : public Operator {
    public:
        ~AdaptiveMaxPoolingOperator() override = default;
        explicit AdaptiveMaxPoolingOperator(uint32_t output_h, uint32_t output_w);
        void set_output_h(uint32_t output_h);
        void set_output_w(uint32_t output_w);
        uint32_t get_output_h();
        uint32_t get_output_w();
    private:
        uint32_t output_h_;
        uint32_t output_w_;
    };
}
#endif //KUIPER_COURSE_ADAPTIVEAVGPOOLING_H
