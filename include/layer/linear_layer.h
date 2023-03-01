//
// Created by zpx on 2023/02/28.
//

#ifndef KUIPER_COURSE_LINEAR_LAYER_H
#define KUIPER_COURSE_LINEAR_LAYER_H

#include "ops/linear_op.h"
#include "layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    class LinearLayer : public Layer {
    public:
        LinearLayer(const std::shared_ptr<Operator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs);
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
    private:
        std::unique_ptr<LinearOperator> op_;
    };
}
#endif //KUIPER_COURSE_LINEAR_LAYER_H
