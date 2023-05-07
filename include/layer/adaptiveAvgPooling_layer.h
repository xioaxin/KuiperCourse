//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_ADAPTIVEAVGPOOLING_LAYER_H
#define KUIPER_COURSE_ADAPTIVEAVGPOOLING_LAYER_H

#include "layer.h"
#include "ops/adaptiveAvgPooling_op.h"

namespace kuiper_infer {
    class AdaptiveAvgPoolingLayer : public Layer {
    public:
        ~AdaptiveAvgPoolingLayer() override = default;
        explicit AdaptiveAvgPoolingLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<AdaptiveAvgPoolingOperator> op_;
    };
}
#endif //KUIPER_COURSE_ADAPTIVEAVGPOOLING_LAYER_H
