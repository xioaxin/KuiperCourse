//
// Created by zpx on 2023/01/01.
//

#ifndef KUIPER_COURSE_RELU6_LAYER_H
#define KUIPER_COURSE_RELU6_LAYER_H

#include "ops/relu6_op.h"
#include "layer_layer.h"

namespace kuiper_infer {
    class Relu6Layer : public Layer {
    public:
        ~Relu6Layer() override = default;
        // 通过这里，把relu_op中的thresh告知给relu layer, 因为计算的时候要用到
        explicit Relu6Layer(const std::shared_ptr<RuntimeOperator> &op);
        // 执行relu 操作的具体函数Forwards
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        void ForwardsCuda(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<Relu6Operator> op_;
    };
}
#endif //KUIPER_COURSE_RELU6_LAYER_H
