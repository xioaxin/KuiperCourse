//
// Created by zpx on 2023/04/19.
//

#ifndef KUIPER_COURSE_OPERATOR_FACTORY_H
#define KUIPER_COURSE_OPERATOR_FACTORY_H

#include "ops/runtime_op.h"
/*
 * static map<type,智能指针>  抽象工程           factory(Procut)           -> porcut: procut1()
 */
namespace kuiper_infer {
    class RuntimeOperatorRegisterer {
    public:
        typedef std::shared_ptr<RuntimeOperator> (*Creator)(const std::string type);
        typedef std::map<OpType, Creator> CreateRegistry;
        static void RegisterCreator(OpType opType, const Creator &creator);
        static std::shared_ptr<RuntimeOperator> CreateRuntimeOperator(const std::string type);
        static CreateRegistry &Registry();
    };

    class RuntimeOperatorRegistererWrapper {
    public:
        explicit RuntimeOperatorRegistererWrapper(OpType OpType, const RuntimeOperatorRegisterer::Creator &creator) {
            RuntimeOperatorRegisterer::RegisterCreator(OpType, creator);
        }
    };
}
#endif //KUIPER_COURSE_OPERATOR_FACTORY_H
