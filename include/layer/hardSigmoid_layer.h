//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_HARDSIGMOID_LAYER_H
#define KUIPER_COURSE_HARDSIGMOID_LAYER_H

#include "layer.h"
#include "factory/layer_factory.hpp"
#include "ops/hardSigmoid_op.h"

namespace kuiper_infer {
    class HardSigmoidLayer : public Layer {
    public:
        explicit HardSigmoidLayer(const std::shared_ptr<RuntimeOperator> &op);
        ~HardSigmoidLayer() override = default;
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<HardSigmoidOperator> op_;
    };
}
#endif //KUIPER_COURSE_HARDSIGMOID_LAYER_H
