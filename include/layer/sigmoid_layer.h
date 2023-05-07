//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_SIGMOID_LAYER_H
#define KUIPER_COURSE_SIGMOID_LAYER_H

#include "layer.h"
#include "ops/sigmoid_op.h"

namespace kuiper_infer {
    class SigmoidLayer : public Layer {
    public:
        ~SigmoidLayer() override = default;
        explicit SigmoidLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<SigmoidOperator> op_;
    };
}
#endif //KUIPER_COURSE_SIGMOID_LAYER_H
