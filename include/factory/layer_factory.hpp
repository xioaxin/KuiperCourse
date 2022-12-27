//
// Created by zpx on 2022/12/26.
//

#ifndef KUIPER_COURSE_LAYER_FACTOR_H
#define KUIPER_COURSE_LAYER_FACTOR_H

#include "ops/ops.h"
#include "layer/layer.h"

namespace kuiper_infer {
    class LayerRegisterer {
    public:
        // TODO: 待理解
        typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op);

        typedef std::map<OpType, Creator> CreateRegistry;

        static void RegisterCreator(OpType opType, const Creator &creator);

        static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op);

        static CreateRegistry &Registry();
    };

    class LayerRegistererWrapper {
    public:
        LayerRegistererWrapper(OpType OpType, const LayerRegisterer::Creator &creator) {
            LayerRegisterer::RegisterCreator(OpType, creator);
        }
    };
}
#endif //KUIPER_COURSE_LAYER_FACTOR_H
