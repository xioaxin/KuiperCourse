//
// Created by zpx on 2023/04/19.
//

#include "ops/runtime_op.h"
#include "factory/operator_factory.h"
#include <glog/logging.h>

namespace kuiper_infer {
    void RuntimeOperatorRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
        CHECK(creator != nullptr) << "Operator creator is empty";
        CreateRegistry &registry = Registry();
        CHECK_EQ(registry.count(op_type), 0) << "Operator type: " << int(op_type) << " has already registered!";
        registry.insert({op_type, creator});
    }

    std::shared_ptr<RuntimeOperator> RuntimeOperatorRegisterer::CreateRuntimeOperator(const std::string type) {
        CreateRegistry &registry = Registry();
        const OpType op_type = PNNX_TO_KUIPER_TABLE[type];
        LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the RuntimeOperator type: " << int(op_type);
        const auto &creator = registry.find(op_type)->second;
        return creator(type);
    }

    // 单例生成模型注册表（哈希表）
    RuntimeOperatorRegisterer::CreateRegistry &RuntimeOperatorRegisterer::Registry() {
        static CreateRegistry *kRegistry = new CreateRegistry();
        CHECK(kRegistry != nullptr) << "Global runtimeOperator register init failed!";
        return *kRegistry;
    }
}