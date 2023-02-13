//
// Created by zpx on 2023/02/13.
//
#include "parser/parse_expression.h"
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "layer/expression_layer.h"

static void showNode(const std::shared_ptr<kuiper_infer::TokenNode> &node) {
    if (node == nullptr)return;
    showNode(node->left_);
    if (node->num_index_ < 0) {
        if (node->num_index_ == -int(kuiper_infer::TokenType::TokenAdd)) {
            LOG(INFO) << "ADD";
        } else if (node->num_index_ == -int(kuiper_infer::TokenType::TokenMul)) {
            LOG(INFO) << "MUL";
        } else if (node->num_index_ == -int(kuiper_infer::TokenType::TokenSub)) {
            LOG(INFO) << "SUB";
        } else if (node->num_index_ == -int(kuiper_infer::TokenType::TokenDiv)) {
            LOG(INFO) << "DIV";
        }
    } else {
        LOG(INFO) << "NUM: " << node->num_index_;
    }
    showNode(node->right_);
}

static void showNodes(const std::vector<std::shared_ptr<kuiper_infer::TokenNode>> &nodes) {
    if (nodes.empty()) {
        return;
    }
    for (auto node: nodes) {
        showNode(node);
    }
}

TEST(test_expression, expression1) {
    using namespace kuiper_infer;
    const std::string &statement = "add(@1,@2)";
    ExpressionParser parser(statement);
    //std::vector<std::shared_ptr<TokenNode>
    const auto &node_tokens = parser.generate();
    showNodes(node_tokens);
}

TEST(test_expression, expression2) {
    using namespace kuiper_infer;
    const std::string &statement = "add(mul(@0,@1),@2)";
    ExpressionParser parser(statement);
    const auto &node_tokens = parser.generate();
    showNodes(node_tokens);
}

TEST(test_expression, expression3) {
    using namespace kuiper_infer;
    const std::string &statement = "add(mul(@0,@1),mul(@2,@3))";
    ExpressionParser parser(statement);
    const auto &node_tokens = parser.generate();
    showNodes(node_tokens);
}

TEST(test_expression, expression4) {
    using namespace kuiper_infer;
    //div在词法、语法解析中都是没有的，你要在两个地方加上去
    const std::string &statement = "add(div(@0,@1),@2)";
    ExpressionParser parser(statement);
    const auto &node_tokens = parser.generate();
    showNodes(node_tokens);
}

TEST(test_expression, add) {
    using namespace kuiper_infer;
    const std::string &expr = "add(@0,@1)";
    std::shared_ptr<ExpressionOperator> expression_op = std::make_shared<ExpressionOperator>(expr);
    ExpressionLayer layer(expression_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    std::vector<std::shared_ptr<ftensor >> outputs;
    int batch_size = 4;
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(1.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(2.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
        outputs.push_back(output);
    }
    layer.Forwards(inputs, outputs);
    for (int i = 0; i < batch_size; ++i) {
        const auto &result = outputs.at(i);
        for (int j = 0; j < result->size(); ++j) {
            ASSERT_EQ(result->index(j), 3.f);
        }
    }
}

TEST(test_expression, sub) {
    using namespace kuiper_infer;
    const std::string &expr = "sub(@0,@1)";
    std::shared_ptr<ExpressionOperator> expression_op = std::make_shared<ExpressionOperator>(expr);
    ExpressionLayer layer(expression_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    std::vector<std::shared_ptr<ftensor >> outputs;
    int batch_size = 4;
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(2.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(1.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
        outputs.push_back(output);
    }
    layer.Forwards(inputs, outputs);
    for (int i = 0; i < batch_size; ++i) {
        const auto &result = outputs.at(i);
        for (int j = 0; j < result->size(); ++j) {
            ASSERT_EQ(result->index(j), 1.f);
        }
    }
}

TEST(test_expression, complex) {
    using namespace kuiper_infer;
    const std::string &expr = "add(mul(@0,@1),@2)";
    std::shared_ptr<ExpressionOperator> expression_op = std::make_shared<ExpressionOperator>(expr);
    ExpressionLayer layer(expression_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    std::vector<std::shared_ptr<ftensor >> outputs;
    int batch_size = 4;
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(1.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(2.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(3.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
        outputs.push_back(output);
    }
    layer.Forwards(inputs, outputs);
    for (int i = 0; i < batch_size; ++i) {
        const auto &result = outputs.at(i);
        for (int j = 0; j < result->size(); ++j) {
            ASSERT_EQ(result->index(j), 5.f);
        }
    }
}

TEST(test_expression, complex_add_mul_sub_div) {
    using namespace kuiper_infer;
    const std::string &expr = "div(sub(add(mul(@0,@1),@2),@3),@4)";
    std::shared_ptr<ExpressionOperator> expression_op = std::make_shared<ExpressionOperator>(expr);
    ExpressionLayer layer(expression_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    std::vector<std::shared_ptr<ftensor >> outputs;
    int batch_size = 4;
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(1.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(2.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(3.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(2.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(3.f);
        inputs.push_back(input);
    }
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
        outputs.push_back(output);
    }
    layer.Forwards(inputs, outputs);
    for (int i = 0; i < batch_size; ++i) {
        const auto &result = outputs.at(i);
        for (int j = 0; j < result->size(); ++j) {
            ASSERT_EQ(result->index(j), 1.f);
        }
    }
}