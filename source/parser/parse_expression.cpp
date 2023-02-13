//
// Created by zpx on 2023/02/11.
//
#include "parser/parse_expression.h"
#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>
#include <glog/logging.h>

namespace kuiper_infer {
    void reversePolish(const std::shared_ptr<TokenNode> &roo_node,
                       std::vector<std::shared_ptr<TokenNode>> &reverse_polish) {
        if (roo_node != nullptr) {
            reversePolish(roo_node->left_, reverse_polish);
            reversePolish(roo_node->right_, reverse_polish);
            reverse_polish.push_back(roo_node);
        }
    }

    void ExpressionParser::Tokenizer(bool need_retoken) {
        if (!need_retoken && !tokens_.empty())return;
        CHECK(!statement_.empty()) << "The input statement is empty!";
        // 清除多余的空格
        statement_.erase(std::remove_if(statement_.begin(), statement_.end(), [](char c) {
            return std::isspace(c);
        }), statement_.end());
        CHECK(!statement_.empty()) << "The input statement is empty!";
        for (int32_t i = 0; i < statement_.size();) {
            char c = statement_.at(i);
            if (c == 'a') {  // 1. 相加 add
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'd')
                                << "Parse add token failed, illegal character: " << c;
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'd')
                                << "Parse add token failed, illegal character: " << c;
                Token token(TokenType::TokenAdd, i, i + 3); // 将操作符入栈
                tokens_.push_back(token);
                std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_operation);
                i = i + 3;
            } else if (c == 'd') { // 2. 相除 div
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'i')
                                << "Parse div token failed, illegal character: " << c;
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'v')
                                << "Parse div token failed, illegal character: " << c;
                Token token(TokenType::TokenDiv, i, i + 3);
                tokens_.push_back(token);
                std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_operation);
                i = i + 3;
            } else if (c == 's') {   // 3. 相减 sub
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
                                << "Parse sub token failed, illegal character: " << c;
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'b')
                                << "Parse sub token failed, illegal character: " << c;
                Token token(TokenType::TokenSub, i, i + 3);
                tokens_.push_back(token);
                std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_operation);
                i = i + 3;
            } else if (c == 'm') { // 4. 相乘 mul
                CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
                                << "Parse add token failed, illegal character: " << c;
                CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
                                << "Parse add token failed, illegal character: " << c;
                Token token(TokenType::TokenMul, i, i + 3);
                tokens_.push_back(token);
                std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
                token_strs_.push_back(token_operation);
                i = i + 3;
            } else if (c == '@') {  // 5. 提取操作数
                CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
                                << "Parse number token failed, illegal character: " << c;
                int32_t j = i + 1;
                for (; j < statement_.size(); ++j) {
                    if (!std::isdigit(statement_.at(j))) {
                        break;
                    }
                }
                Token token(TokenType::TokenInputNumber, i, j);
                CHECK(token.start_pos_ < token.end_pos_);
                tokens_.push_back(token);
                std::string token_input_number = std::string(statement_.begin() + i, statement_.begin() + j);
                token_strs_.push_back(token_input_number);
                i = j;
            } else if (c == ',') {
                Token token(TokenType::TokenComma, i, i + 1);
                tokens_.push_back(token);
                std::string token_comma = std::string(statement_.begin() + i, statement_.begin() + i + 1);
                token_strs_.push_back(token_comma);
                i += 1;
            } else if (c == '(') {
                Token token(TokenType::TokenLeftBracket, i, i + 1);
                tokens_.push_back(token);
                std::string token_left_bracket = std::string(statement_.begin() + i, statement_.begin() + i + 1);
                token_strs_.push_back(token_left_bracket);
                i += 1;
            } else if (c == ')') {
                Token token(TokenType::TokenRightBracket, i, i + 1);
                tokens_.push_back(token);
                std::string token_right_bracket = std::string(statement_.begin() + i, statement_.begin() + i + 1);
                token_strs_.push_back(token_right_bracket);
                i += 1;
            } else {
                LOG(FATAL) << "Unknown  illegal character: " << c;
            }
        }
    }

    const std::vector<Token> &ExpressionParser::tokens() const {
        return this->tokens_;
    }

    const std::vector<std::string> &ExpressionParser::token_strs() const {
        return this->token_strs_;
    }

    std::shared_ptr<TokenNode> ExpressionParser::generate_(int32_t &index) {
        CHECK(index < this->tokens_.size());
        const auto current_token = this->tokens_.at(index);
        CHECK(current_token.tokenType_ == TokenType::TokenInputNumber ||
              current_token.tokenType_ == TokenType::TokenAdd ||
              current_token.tokenType_ == TokenType::TokenMul ||
              current_token.tokenType_ == TokenType::TokenSub ||
              current_token.tokenType_ == TokenType::TokenDiv);
        if (current_token.tokenType_ == TokenType::TokenInputNumber) {
            uint32_t start_pos = current_token.start_pos_ + 1;
            uint32_t end_pos = current_token.end_pos_;
            CHECK(end_pos > start_pos);
            CHECK(end_pos <= this->statement_.length());
            const std::string &str_number =
                    std::string(this->statement_.begin() + start_pos, this->statement_.begin() + end_pos);
            return std::make_shared<TokenNode>(std::stoi(str_number), nullptr, nullptr);
        } else if (current_token.tokenType_ == TokenType::TokenMul || current_token.tokenType_ == TokenType::TokenAdd ||
                   current_token.tokenType_ == TokenType::TokenSub || current_token.tokenType_ == TokenType::TokenDiv) {
            std::shared_ptr<TokenNode> current_node = std::make_shared<TokenNode>();
            current_node->num_index_ = -int(current_token.tokenType_);
            index += 1;
            CHECK(index < this->tokens_.size());
            CHECK(this->tokens_.at(index).tokenType_ == TokenType::TokenLeftBracket);
            index += 1;
            CHECK(index < this->tokens_.size());
            const auto left_token = this->tokens_.at(index);
            if (left_token.tokenType_ == TokenType::TokenInputNumber
                || left_token.tokenType_ == TokenType::TokenAdd || left_token.tokenType_ == TokenType::TokenMul ||
                left_token.tokenType_ == TokenType::TokenSub || left_token.tokenType_ == TokenType::TokenDiv) {
                current_node->left_ = generate_(index);
            } else {
                LOG(FATAL) << "Unknown token type: " << int(left_token.tokenType_);
            }
            index += 1;
            CHECK(index < this->tokens_.size());
            CHECK(this->tokens_.at(index).tokenType_ == TokenType::TokenComma);
            index += 1;
            CHECK(index < this->tokens_.size());
            const auto right_token = this->tokens_.at(index);
            if (right_token.tokenType_ == TokenType::TokenInputNumber
                || right_token.tokenType_ == TokenType::TokenAdd || right_token.tokenType_ == TokenType::TokenMul ||
                left_token.tokenType_ == TokenType::TokenSub || left_token.tokenType_ == TokenType::TokenDiv) {
                current_node->right_ = generate_(index);
            } else {
                LOG(FATAL) << "Unknown token type: " << int(left_token.tokenType_);
            }
            //负的都是操作数
            index += 1;
            CHECK(index < this->tokens_.size());
            CHECK(this->tokens_.at(index).tokenType_ == TokenType::TokenRightBracket);
            return current_node;
        } else {
            LOG(FATAL) << "Unknown token type: " << int(current_token.tokenType_);
        }
    }

    std::vector<std::shared_ptr<TokenNode>> ExpressionParser::generate() {
        if (this->tokens_.empty()) { // 解析操作数和操作符，分别压入对应的栈中
            this->Tokenizer(true);
        }
        int index = 0;
        std::shared_ptr<TokenNode> root = generate_(index);
        CHECK(root != nullptr);
        CHECK(index == tokens_.size() - 1);
        std::vector<std::shared_ptr<TokenNode>> reverse_polish;
        reversePolish(root, reverse_polish);
        return reverse_polish;
    }
}