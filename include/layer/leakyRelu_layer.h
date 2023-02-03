//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_LEAKYRELU_LAYER_H
#define KUIPER_COURSE_LEAKYRELU_LAYER_H

#include "layer.h"
#include "ops/leakyRelu_op.h"

namespace kuiper_infer {
    class LeakyReluLayer : public Layer {
    public:
        ~LeakyReluLayer() override = default;
        // 通过这里，把relu_op中的thresh告知给relu layer, 因为计算的时候要用到
        explicit LeakyReluLayer(const std::shared_ptr<Operator> &op);
        // 执行relu 操作的具体函数Forwards
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        // 下节的内容，不用管
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
    private:
        std::unique_ptr<LeakyReluOperator> op_;
    };
}
#endif //KUIPER_COURSE_LEAKYRELU_LAYER_H
