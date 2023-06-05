//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_ADAPTIVEAVGPOOLING_LAYER_H
#define KUIPER_COURSE_ADAPTIVEAVGPOOLING_LAYER_H

#include "layer_layer.h"
#include "ops/adaptiveMaxPooling_op.h"

namespace kuiper_infer {
    class AdaptiveMaxPoolingLayer : public Layer {
    public:
        explicit AdaptiveMaxPoolingLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        void ForwardsCuda(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<float>>>&outputs) override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<AdaptiveMaxPoolingOperator> op_;
    };
}
#endif //KUIPER_COURSE_ADAPTIVEAVGPOOLING_LAYER_H
