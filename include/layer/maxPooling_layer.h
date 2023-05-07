//
// Created by zpx on 2023/02/03.
//

#ifndef KUIPER_COURSE_MAXPOOLING_LAYER_H
#define KUIPER_COURSE_MAXPOOLING_LAYER_H

#include "layer.h"
#include "ops/maxPooling_op.h"

namespace kuiper_infer {
    class MaxPoolingLayer : public Layer {
    public:
        explicit MaxPoolingLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<MaxPoolingOperator> op_;
    };
}
#endif //KUIPER_COURSE_MAXPOOLING_LAYER_H
