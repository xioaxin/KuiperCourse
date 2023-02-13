//
// Created by zpx on 2023/02/12.
//

#ifndef KUIPER_COURSE_EXPRESSION_OP_H
#define KUIPER_COURSE_EXPRESSION_OP_H

#include <vector>
#include <string>
#include <memory>
#include "ops.h"
#include "parser/parse_expression.h"

namespace kuiper_infer {
    class ExpressionOperator : public Operator {
    public:
        explicit ExpressionOperator(const std::string &expr);
        std::vector<std::shared_ptr<TokenNode>> generate();
    private:
        std::shared_ptr<ExpressionParser> parser_;
        std::vector<std::shared_ptr<TokenNode>> nodes_;
        std::string expr_;
    };
}
#endif //KUIPER_COURSE_EXPRESSION_OP_H
