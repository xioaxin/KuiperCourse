//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_FLATTEN_LAYER_H
#define KUIPER_COURSE_FLATTEN_LAYER_H

#include "layer.h"
#include "factory/layer_factory.hpp"
#include "ops/flatten_op.h"

namespace kuiper_infer {
    class FlattenLayer : public Layer {
    public:
        FlattenLayer(const std::shared_ptr<Operator> &op);
        ~FlattenLayer() override = default;
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs);
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
    private:
        std::unique_ptr<FlattenOperator> op_;
    };
}
#endif //KUIPER_COURSE_FLATTEN_LAYER_H
