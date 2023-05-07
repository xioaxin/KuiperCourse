//
// Created by zpx on 2023/04/01.
//
#include "ops/runtime_op.h"

namespace kuiper_infer {
    RuntimeOperator::RuntimeOperator() : op_type_(OpType::kOperatorUnknown) {}

    RuntimeOperator::~RuntimeOperator() {
        for (auto &item: this->params) {
            if (item.second->type_ != RuntimeParameterType::kParameterDelete) {
                item.second->type_ = RuntimeParameterType::kParameterDelete;
            } else {
                delete item.second;
            }
        }
    }

    RuntimeOperator::RuntimeOperator(OpType opType) : op_type_(opType) {}

    void RuntimeOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
    }

    void RuntimeOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

//    RuntimeOperator::RuntimeOperator(const kuiper_infer::RuntimeOperator &runtimeOperator) {
//        this->op_type_ = runtimeOperator.op_type_;
//        this->name = runtimeOperator.name;
//        std::copy(runtimeOperator.params.begin(), runtimeOperator.params.end(), std::inserter(params, params.begin()));
//        std::copy(runtimeOperator.attribute.begin(), runtimeOperator.attribute.end(), std::inserter(attribute, attribute.begin()));
//    }
}