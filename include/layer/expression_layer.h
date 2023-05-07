//
// Created by zpx on 2023/02/12.
//

#ifndef KUIPER_COURSE_EXPRESSION_LAYER_H
#define KUIPER_COURSE_EXPRESSION_LAYER_H

#include "layer.h"
#include "ops/expression_op.h"

namespace kuiper_infer {
    class ExpressionLayer : public Layer {
    public:
        explicit ExpressionLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<ExpressionOperator> op_;
    };
}
#endif //KUIPER_COURSE_EXPRESSION_LAYER_H
