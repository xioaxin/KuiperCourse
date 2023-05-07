//
// Created by zpx on 2023/02/26.
//
#include "ops/flatten_op.h"

namespace kuiper_infer {
    FlattenOperator::FlattenOperator() : RuntimeOperator(OpType::kOperatorFlatten) {}

    FlattenOperator::FlattenOperator(int start_dim, int end_dim) :
            RuntimeOperator(OpType::kOperatorFlatten), start_dim_(start_dim), end_dim_(end_dim) {}

    void FlattenOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->start_dim_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("start_dim"))->value;
        this->end_dim_ = dynamic_cast<RuntimeParameterInt *>(runtimeParameter.at("end_dim"))->value;
    }

    void FlattenOperator::initialAttribute(
            const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> FlattenOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorFlatten);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<FlattenOperator>();
        return runtimeOperator;
    }

    int FlattenOperator::getStartDim() const {
        return start_dim_;
    }

    void FlattenOperator::setStartDim(int startDim) {
        start_dim_ = startDim;
    }

    int FlattenOperator::getEndDim() const {
        return end_dim_;
    }

    void FlattenOperator::setEndDim(int endDim) {
        end_dim_ = endDim;
    }

    RuntimeOperatorRegistererWrapper kFlattenOperator(OpType::kOperatorFlatten, FlattenOperator::CreateInstance);
}

