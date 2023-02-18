//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_IR_H
#define KUIPER_COURSE_RUNTIME_IR_H

#include <vector>
#include <string>
#include <glog/logging.h>
#include <memory>
#include <map>
#include <queue>
#include "ir.h"
#include "factory/layer_factory.hpp"
#include "runtime/runtime_operand.h"
#include "runtime_op.h"

namespace kuiper_infer {
    class RuntimeGraphShape {
    public:
        static void initOperatorInputTensor(const std::vector<std::shared_ptr<RuntimeOperator>> &operators);
        static void initOperatorOutputTensor(const std::vector<pnnx::Operator *> &pnnx_operators,
                                             const std::vector<std::shared_ptr<RuntimeOperator>> &operators);
    };

    class RuntimeGraph {
    public:
        bool init(); //判断是否已经初始化计算图
        void build(const std::string &input_name, const std::string &output_name); //构建计算图
        RuntimeGraph(std::string param_path, std::string bin_path); //初始化计算图
        void set_bin_path(const std::string &bin_path);
        void set_param_path(const std::string &param_path);
        const std::string &param_path() const;
        const std::string &bin_path() const;
        const std::vector<std::shared_ptr<RuntimeOperator>> operators() const;
    private:
        // 初始化输入节点的操数
        static void initInputOperators(const std::vector<pnnx::Operand *> &inputs,
                                       const std::shared_ptr<RuntimeOperator> &runtime_operator);
        // 初始化输出节点的操作数
        static void initOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                                        const std::shared_ptr<RuntimeOperator> &runtime_operator);
        // 初始化计算图节点的属性
        static void initGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                   const std::shared_ptr<RuntimeOperator> &runtime_operator);
        // 初始化计算节点的参数
        static void initGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                    const std::shared_ptr<RuntimeOperator> &runtime_operator);
    private:
        enum class GraphState {
            NeedInit = -2,
            NeedBuild = -1,
            Complete = 0,
        };
        GraphState graphState_ = GraphState::NeedInit;
        std::string input_name_;
        std::string output_name_;
        std::string param_path_;
        std::string bin_path_;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> input_operators_maps_;
        std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_maps_;
        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
        std::unique_ptr<pnnx::Graph> graph_;
    };
}
#endif //KUIPER_COURSE_RUNTIME_IR_H
