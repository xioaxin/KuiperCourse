//
// Created by zpx on 2023/05/07.
//

#ifndef KUIPER_COURSE_VIEW_LAYER_H
#define KUIPER_COURSE_VIEW_LAYER_H

#include "layer.h"
#include "ops/view_op.h"

namespace kuiper_infer {
    class ViewLayer : public Layer {
    public:
        explicit ViewLayer(const std::shared_ptr<RuntimeOperator> &op);
        void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
        void Forwards() override;
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<RuntimeOperator> &op);
    private:
        std::unique_ptr<ViewOperator> op_;
    };
}
#endif //KUIPER_COURSE_VIEW_LAYER_H
