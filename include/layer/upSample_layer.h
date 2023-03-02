//
// Created by zpx on 2023/03/01.
//

#ifndef KUIPER_COURSE_UPSAMPLE_LAYER_H
#define KUIPER_COURSE_UPSAMPLE_LAYER_H

#include "layer.h"
#include "factory/layer_factory.hpp"
#include "ops/upSample_op.h"

namespace kuiper_infer {
    class UpSampleLayer : public Layer {
    public:
        explicit UpSampleLayer(const std::shared_ptr<Operator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs);
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
    private:
        std::unique_ptr<UpSampleOperator> op_;
    };
}
#endif //KUIPER_COURSE_UPSAMPLE_LAYER_H
