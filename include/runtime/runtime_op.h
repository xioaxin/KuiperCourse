//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_OP_H
#define KUIPER_COURSE_RUNTIME_OP_H

#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include "factory/layer_factory.hpp"
#include "runtime_operand.h"
#include "runtime_attr.h"
#include "runtime_parameter.h"

namespace kuiper_infer {
    class Layer;

    struct RuntimeOperator {
        int32_t meet_num = 0;

        ~RuntimeOperator() {
            for (const auto &param: this->params) {
                delete param.second;
            }
        }

        std::string name;                       // 计算节点名称
        std::string type;                       // 计算节点类型
        std::shared_ptr<Layer> layer;   // 节点对应的计算Layer
        std::vector<std::string> output_names;  // 输出名称
        std::shared_ptr<RuntimeOperand> output_operands; //输出操作数
        std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
        std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;
        std::map<std::string, RuntimeParameter *> params;
        std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
    };
}
#endif //KUIPER_COURSE_RUNTIME_OP_H
