//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_CAT_H
#define KUIPER_COURSE_CAT_H

#include "layer.h"
#include "factory/layer_factory.hpp"
#include "ops/cat_op.h"

namespace kuiper_infer {
    class CatLayer : public Layer {
    public:
        explicit CatLayer(const std::shared_ptr<Operator> &op);
        void Forwards(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs);
        static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);
    private:
        std::unique_ptr<CatOperator> op_;
    };
}
#endif //KUIPER_COURSE_CAT_H
