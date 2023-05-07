// Reference: https://paperswithcode.com/method/hard-swish
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_HARDSWISH_LAYER_H
#define KUIPER_COURSE_HARDSWISH_LAYER_H

#include "ops/hardSwish_op.h"
#include "layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    class HardSwishLayer : public Layer {
    public:
        explicit HardSwishLayer(const std::shared_ptr<RuntimeOperator> &op);
        ~HardSwishLayer() override = default;
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs);
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<HardSwishOperator> op_;
    };
}
#endif //KUIPER_COURSE_HARDSWISH_LAYER_H
