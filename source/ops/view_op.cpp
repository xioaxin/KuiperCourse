//
// Created by zpx on 2023/05/07.
//
#include "ops/view_op.h"

namespace kuiper_infer {
    ViewOperator::ViewOperator() : RuntimeOperator(OpType::kOperatorView) {}

    ViewOperator::ViewOperator(std::vector<int> shape) : RuntimeOperator(OpType::kOperatorView), shape_(shape) {}

    void ViewOperator::initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->shape_ = dynamic_cast<RuntimeParameterIntArray *>(runtimeParameter.at("shape").get())->value;
    }

    void ViewOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> ViewOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorView);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<ViewOperator>();
        return runtimeOperator;
    }

    void ViewOperator::setShape(const std::vector<int> shape) {
        this->shape_ = shape;
    }

    std::vector<int> ViewOperator::getShape() const {
        return this->shape_;
    }

    RuntimeOperatorRegistererWrapper kViewOperator(OpType::kOperatorView, ViewOperator::CreateInstance);
}

