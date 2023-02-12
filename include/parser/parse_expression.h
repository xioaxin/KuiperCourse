//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_PARSE_EXPRESSION_H
#define KUIPER_COURSE_PARSE_EXPRESSION_H

#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace kuiper_infer {
    enum class TokenType {
        TokenUnknown = -1,
        TokenInputNumber = 0,
        TokenComma = 1,
        TokenAdd = 2,
        TokenMul = 3,
        TokenLeftBracket = 4,
        TokenRightBracket = 5,
    };

    struct Token {
        TokenType tokenType_ = TokenType::TokenUnknown;
        int32_t start_pos_ = 0;
        int32_t end_pos_ = 0;

        Token(TokenType toke_type, int32_t start_pos, int32_t end_pos) :
                tokenType_(toke_type), start_pos_(start_pos), end_pos_(end_pos) {}
    };

    class TokenNode {
    public:
        int32_t num_index_ = -1;
        std::shared_ptr<TokenNode> left_ = nullptr;
        std::shared_ptr<TokenNode> right_ = nullptr;

        TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right) :
                num_index_(num_index), left_(left), right_(right) {};
        TokenNode()=default;
    };

    class ExpressionParser {
    public:
        explicit ExpressionParser(std::string statement) : statement_(statement) {};
        void Tokenizer(bool need_retoken = false);
        std::shared_ptr<TokenNode> generate();
        const std::vector<Token> &tokens() const;
        const std::vector<std::string> &token_strs() const;
    private:
        std::shared_ptr<TokenNode> generate_(int32_t &index);
        std::vector<Token> tokens_;
        std::vector<std::string> token_strs_;
        std::string statement_;
    };
}
#endif //KUIPER_COURSE_PARSE_EXPRESSION_H
