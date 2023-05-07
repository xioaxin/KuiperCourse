//
// Created by zpx on 2022/12/26.
//

#ifndef KUIPER_COURSE_LAYER_FACTOR_H
#define KUIPER_COURSE_LAYER_FACTOR_H

#include "layer/layer.h"
#include "ops/runtime_op.h"

namespace kuiper_infer {
    enum class OpType;

    class RuntimeOperator;

    class LayerRegisterer {
    public:
        // TODO: 待理解
        typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<RuntimeOperator> &op);
        typedef std::map<OpType, Creator> CreateRegistry;
        static void RegisterCreator(OpType opType, const Creator &creator);
        static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator> &op);
        static CreateRegistry &Registry();
    };

    class LayerRegistererWrapper {
    public:
        explicit LayerRegistererWrapper(OpType OpType, const LayerRegisterer::Creator &creator) {
            LayerRegisterer::RegisterCreator(OpType, creator);
        }
    };
}
#endif //KUIPER_COURSE_LAYER_FACTOR_H
