//
// Created by zpx on 2023/02/13.
//
#include <glog/logging.h>
#include "ops/expression_op.h"

namespace kuiper_infer {
    ExpressionOperator::ExpressionOperator(const std::string &expr) :
            Operator(OpType::kOperatorExpression), expr_(expr) {
        this->parser_ = std::make_shared<ExpressionParser>(this->expr_);
    }

    std::vector<std::shared_ptr<TokenNode>> ExpressionOperator::generate() {
        CHECK(this->parser_ != nullptr);
        this->nodes_ = this->parser_->generate();
        return this->nodes_;
    }
}