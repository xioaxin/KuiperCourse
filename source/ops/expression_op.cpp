//
// Created by zpx on 2023/02/13.
//
#include <glog/logging.h>
#include "ops/expression_op.h"

namespace kuiper_infer {
    ExpressionOperator::ExpressionOperator() : RuntimeOperator(OpType::kOperatorExpression) {}

    ExpressionOperator::ExpressionOperator(const std::string &expr) : RuntimeOperator(OpType::kOperatorExpression),
                                                                      expr_(expr) {
        this->parser_ = std::make_shared<ExpressionParser>(this->expr_);
    }

    std::vector<std::shared_ptr<TokenNode>> ExpressionOperator::generate() {
        CHECK(this->parser_ != nullptr);
        this->nodes_ = this->parser_->generate();
        return this->nodes_;
    }

    void ExpressionOperator::initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) {
        CHECK(!runtimeParameter.empty()) << "The parameter of " << type << "is empty";
        this->expr_ = dynamic_cast<RuntimeParameterString *>(runtimeParameter.at("expr"))->value;
        this->parser_ = std::make_shared<ExpressionParser>(this->expr_);
    }

    void ExpressionOperator::initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) {
    }

    std::shared_ptr<RuntimeOperator> ExpressionOperator::CreateInstance(const std::string type) {
        CHECK(PNNX_TO_KUIPER_TABLE[type] == OpType::kOperatorExpression);
        std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<ExpressionOperator>();
        return runtimeOperator;
    }

    RuntimeOperatorRegistererWrapper kExpressionOperator(OpType::kOperatorExpression, ExpressionOperator::CreateInstance);
}