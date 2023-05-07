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
#include "runtime/runtime_operand.h"
#include "runtime/runtime_attr.h"
#include "runtime/runtime_parameter.h"

namespace kuiper_infer {
    enum class OpType {
        kOperatorUnknown = -1,
        kOperatorRelu = 0,
        kOperatorSigmoid = 1,
        kOperatorRelu6 = 2,
        kOperatorLeakyRelu = 3,
        kOperatorMaxPooling = 4,
        kOperatorExpression = 5,
        kOperatorAvgPooling = 6,
        kOperatorAdaptiveAvgPooling = 7,
        kOperatorAdaptiveMaxPooling = 8,
        kOperatorConvolution = 9,
        kOperatorBatchNorm = 10,
        kOperatorSoftMax = 11,
        kOperatorCat = 12,
        kOperatorFlatten = 13,
        kOperatorHardSigmoid = 14,
        kOperatorHardSwish = 15,
        kOperatorSilu = 16,
        kOperatorLinear = 17,
        kOperatorUpSample = 18,
        kOperatorInput = 19,
        kOperatorOutput = 20,
    };
    static std::unordered_map<std::string, kuiper_infer::OpType> PNNX_TO_KUIPER_TABLE = {
            {"nn.Conv2d",            OpType::kOperatorConvolution},
            {"nn.ReLU",              OpType::kOperatorRelu},
            {"nn.Linear",            OpType::kOperatorLinear},
            {"nn.MaxPool2d",         OpType::kOperatorMaxPooling},
            {"nn.AdaptiveAvgPool2d", OpType::kOperatorAdaptiveAvgPooling},
            {"pnnx.Input",           OpType::kOperatorInput},
            {"pnnx.Output",          OpType::kOperatorOutput},
            {"pnnx.Expression",      OpType::kOperatorExpression},
            {"torch.flatten",        OpType::kOperatorFlatten},
            {"Tensor.view",          OpType::kOperatorFlatten},
    };

    class RuntimeOperator {
    public:
        int32_t meet_num = 0;
        RuntimeOperator();
        RuntimeOperator(OpType opType);

        virtual ~RuntimeOperator();
        OpType op_type_ = OpType::kOperatorUnknown;
        std::string name;                       // 计算节点名称
        std::string type;                       // 计算节点类型
        std::shared_ptr<Layer> layer;
        std::vector<std::string> output_names;  // 输出名称
        std::shared_ptr<RuntimeOperand> output_operands; //输出操作数
        std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;
        std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;
        std::map<std::string, RuntimeParameter *> params;
        std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
        virtual void initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) = 0;
        virtual void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) = 0;
//        RuntimeOperator(const RuntimeOperator &runtimeOperator);
    };
}
#endif //KUIPER_COURSE_RUNTIME_OP_H
