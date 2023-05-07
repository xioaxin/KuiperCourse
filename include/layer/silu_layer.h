//
// Created by zpx on 2023/02/28.
//

#ifndef KUIPER_COURSE_SILU_LAYER_H
#define KUIPER_COURSE_SILU_LAYER_H

#include "ops/silu_op.h"
#include "layer.h"

namespace kuiper_infer {
    class SiluLayer : public Layer {
    public:
        SiluLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs)override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<SiluOperator> op_;
    };
}
#endif //KUIPER_COURSE_SILU_LAYER_H
