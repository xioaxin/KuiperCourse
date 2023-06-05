//
// Created by zpx on 2023/02/13.
//
#include <glog/logging.h>
#include <stack>
#include "ops/expression_op.h"
#include "layer/layer_layer.h"
#include "layer/expression_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    ExpressionLayer::ExpressionLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("Expression") {
        CHECK(op->op_type_ == OpType::kOperatorExpression) << "Operator was a wrong type: " << int(op->op_type_);
        ExpressionOperator *expressionOperator = dynamic_cast<ExpressionOperator *>(op.get());
        CHECK(expressionOperator != nullptr) << "Expression operator is empty";
        this->op_ = std::make_unique<ExpressionOperator>(*expressionOperator);
    }

    void ExpressionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(!inputs.empty());
        const uint32_t batch_size = outputs.size();
        CHECK(batch_size != 0);
        for (uint32_t i = 0; i < batch_size; i++) { // 构造一个输出矩阵
            CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
            outputs.at(i)->fill(0.f);
        }
        CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorExpression);
        std::stack<std::vector<std::shared_ptr<Tensor<float >>>> op_stack;
        const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->op_->generate();
        for (const auto &token_node: token_nodes) {
            if (token_node->num_index_ >= 0) {
                uint32_t start_pos = token_node->num_index_ * batch_size;
                std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
                for (uint32_t i = 0; i < batch_size; i++) {
                    CHECK(i + start_pos < inputs.size());
                    input_token_nodes.push_back(inputs.at(i + start_pos));
                }
                op_stack.push(input_token_nodes);
            } else {
                const int32_t op = token_node->num_index_;
                CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
                std::vector<std::shared_ptr<Tensor<float>>> input_node1 = op_stack.top();
                CHECK(input_node1.size() == batch_size);
                op_stack.pop();
                std::vector<std::shared_ptr<Tensor<float>>> input_node2 = op_stack.top();
                CHECK(input_node2.size() == batch_size);
                op_stack.pop();
                CHECK(input_node1.size() == input_node2.size());
                std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
//                CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
                for (uint32_t i = 0; i < batch_size; ++i) {
                    if (op == -int(TokenType::TokenAdd)) {
                        output_token_nodes.at(i) = ftensor::elementAdd(input_node1.at(i), input_node2.at(i));
                    } else if (op == -int(TokenType::TokenMul)) {
                        output_token_nodes.at(i) = ftensor::elementMultiply(input_node1.at(i), input_node2.at(i));
                    } else if (op == -int(TokenType::TokenSub)) {
                        output_token_nodes.at(i) = ftensor::elementSub(input_node1.at(i), input_node2.at(i));
                    } else if (op == -int(TokenType::TokenDiv)) {
                        output_token_nodes.at(i) = ftensor::elementDiv(input_node1.at(i), input_node2.at(i));
                    } else {
                        LOG(FATAL) << "Unknown operator type: " << op;
                    }
                }
                op_stack.push(output_token_nodes);
            }
        }
        CHECK(op_stack.size() == 1);
        std::vector<std::shared_ptr<Tensor<float>>> output_node = op_stack.top();
        op_stack.pop();
#pragma omp parallel for num_threads(batch_size)
        for (int i = 0; i < batch_size; ++i) {
            CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
            outputs.at(i) = output_node.at(i);
        }
    }

    std::shared_ptr<Layer> ExpressionLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> expressionLayer = std::make_shared<ExpressionLayer>(op);
        return expressionLayer;
    }
    void ExpressionLayer::Forwards() {
        const std::vector<std::shared_ptr<RuntimeOperand>> &input_operand_datas = this->op_->input_operands_seq;
        std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
        for (const auto &input_operand_data: input_operand_datas) {
            for (const auto &input_data: input_operand_data->datas) {
                layer_input_datas.push_back(input_data);
            }
        }
        CHECK(!layer_input_datas.empty()) << this->op_->name << " Layer input data is empty";
        CHECK(this->op_->output_operands != nullptr && !this->op_->output_operands->datas.empty())
                        << "Layer output data is empty";
        Forwards(layer_input_datas, this->op_->output_operands->datas);
    }
    LayerRegistererWrapper expressionRegisterWrapper(OpType::kOperatorExpression, ExpressionLayer::CreateInstance);
}