//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_SOFTMAX_LAYER_H
#define KUIPER_COURSE_SOFTMAX_LAYER_H

#include "layer.h"
#include "ops/softMax_op.h"
#include <vector>

namespace kuiper_infer {
    class SoftMaxLayer : public Layer {
    public:
        ~SoftMaxLayer() override = default;
        explicit SoftMaxLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs)override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<SoftMaxOperator> op_;
    };
}
#endif //KUIPER_COURSE_SOFTMAX_LAYER_H
