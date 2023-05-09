//
// Created by zpx on 2023/02/12.
//

#ifndef KUIPER_COURSE_EXPRESSION_OP_H
#define KUIPER_COURSE_EXPRESSION_OP_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "parser/parse_expression.h"
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class ExpressionOperator : public RuntimeOperator {
    public:
        ExpressionOperator();
        explicit ExpressionOperator(const std::string &expr);

        ~ExpressionOperator() {};
        std::vector<std::shared_ptr<TokenNode>> generate();
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        std::shared_ptr<ExpressionParser> parser_;
        std::vector<std::shared_ptr<TokenNode>> nodes_;
        std::string expr_;
    };
}
#endif //KUIPER_COURSE_EXPRESSION_OP_H
