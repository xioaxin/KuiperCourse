//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_BATCHNORM_LAYER_H
#define KUIPER_COURSE_BATCHNORM_LAYER_H

#include "layer.h"
#include "ops/batchNorm_op.h"

namespace kuiper_infer {
    class BatchNormLayer : public Layer {
    public:
        explicit BatchNormLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<ftensor>> &inputs,
                     std::vector<std::shared_ptr<ftensor>> &outputs);
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::shared_ptr<BatchNormOperator> op_;
    };
}
#endif //KUIPER_COURSE_BATCHNORM_LAYER_H
