//
// Created by zpx on 2023/02/11.
//
#include "runtime/runtime_ir.h"
#include<memory>
#include <iostream>
#include <queue>
#include <deque>
#include <utility>
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path) :
            param_path_(param_path), bin_path_(bin_path) {};

    void RuntimeGraph::set_bin_path(const std::string &bin_path) {
        bin_path_ = bin_path;
    }

    void RuntimeGraph::set_param_path(const std::string &param_path) {
        param_path_ = param_path;
    }

    const std::string &RuntimeGraph::param_path() const {
        return param_path_;
    }

    const std::string &RuntimeGraph::bin_path() const {
        return bin_path_;
    }

    bool RuntimeGraph::init() {
        if (bin_path_.empty() || param_path_.empty()) {
            LOG(ERROR) << "The bin path or param path is empty";
            return false;
        }
        this->graph_ = std::make_unique<pnnx::Graph>(); // 定义计算图
        int load_result = this->graph_->load(param_path_, bin_path_); // 加载模型属性文件和参数文件
        if (load_result != 0) {
            LOG(ERROR) << "Load param path and bin path error: " << param_path_ << " " << bin_path_;
            return false;
        }
        std::vector<pnnx::Operator *> operators = this->graph_->ops;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }
        this->operators_.clear();
        for (const pnnx::Operator *op: operators) {
            if (!op) {
                LOG(ERROR) << "meet the empty node";
                continue;
            } else {
                std::shared_ptr<RuntimeOperator> runtimeOperator = std::make_shared<RuntimeOperator>();
                // 初始化算子名称和类型
                runtimeOperator->name = op->name;
                runtimeOperator->type = op->type;
                // 初始化算子的input
                const std::vector<pnnx::Operand *> &inputs = op->inputs;
                if (!inputs.empty()) initInputOperators(inputs, runtimeOperator);
                const std::vector<pnnx::Operand *> &outputs = op->outputs;
                if (!outputs.empty())initOutputOperators(outputs, runtimeOperator);
                const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
                if (!attrs.empty())initGraphAttrs(attrs, runtimeOperator);
                const std::map<std::string, pnnx::Parameter> &params = op->params;
                if (!params.empty())initGraphParams(params, runtimeOperator);
                this->operators_.push_back(runtimeOperator);
            }
        }
        graphState_ = GraphState::NeedBuild;
        return true;
    }

    void RuntimeGraph::initInputOperators(const std::vector<pnnx::Operand *> &inputs,
                                          const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *input: inputs) {
            if (!input)continue;
            const pnnx::Operator *producer = input->producer;
            std::shared_ptr<RuntimeOperand> runtimeOperand = std::make_shared<RuntimeOperand>();
            runtimeOperand->name = producer->name;
            runtimeOperand->shapes = input->shape;
            switch (input->type) {
                case 1:
                    runtimeOperand->type = RuntimeDataType::KTypeFloat32;
                    break;
                case 0:
                    runtimeOperand->type = RuntimeDataType::KTypeUnknown;
                    break;
                default:
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
            }
            runtime_operator->input_operands.insert({producer->name, runtimeOperand});
            runtime_operator->input_operands_seq.push_back(runtimeOperand);
        }
    }

    void RuntimeGraph::initOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                                           const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *output: outputs) {
            if (!output)continue;
            const auto &consumer = output->consumers;
            for (const auto &c: consumer) {
                runtime_operator->output_names.push_back(c->name);
            }
        }
    }

    void RuntimeGraph::initGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                       const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &pair: params) {
            const std::string &name = pair.first;
            const pnnx::Parameter &parameter = pair.second;
            const int type = parameter.type;
            switch (type) {
                case int(RuntimeParameterType::kParameterUnknown): {
                    RuntimeParameter *runtime_parameter = new RuntimeParameter();
                    runtime_operator->params.insert({name, runtime_parameter});
                }
                    break;
                case int(RuntimeParameterType::kParameterBool): {
                    RuntimeParameterBool *runtimeParameterBool = new RuntimeParameterBool();
                    runtimeParameterBool->value = parameter.b;
                    runtime_operator->params.insert({name, runtimeParameterBool});
                }
                    break;
                case int(RuntimeParameterType::kParameterInt): {
                    RuntimeParameterInt *runtimeParameterInt = new RuntimeParameterInt();
                    runtimeParameterInt->value = parameter.i;
                    runtime_operator->params.insert({name, runtimeParameterInt});
                }
                    break;
                case int(RuntimeParameterType::kParameterFloat): {
                    RuntimeParameterFloat *runtimeParameterFloat = new RuntimeParameterFloat();
                    runtimeParameterFloat->value = parameter.f;
                    runtime_operator->params.insert({name, runtimeParameterFloat});
                }
                    break;
                case int(RuntimeParameterType::kParameterString): {
                    RuntimeParameterString *runtimeParameterString = new RuntimeParameterString();
                    runtimeParameterString->value = parameter.s;
                    runtime_operator->params.insert({name, runtimeParameterString});
                }
                    break;
                case int(RuntimeParameterType::kParameterIntArray): {
                    RuntimeParameterIntArray *runtimeParameterIntArray = new RuntimeParameterIntArray();
                    runtimeParameterIntArray->value = parameter.ai;
                    runtime_operator->params.insert({name, runtimeParameterIntArray});
                }
                    break;
                case int(RuntimeParameterType::kParameterFloatArray): {
                    RuntimeParameterFloatArray *runtimeParameterFloatArray = new RuntimeParameterFloatArray();
                    runtimeParameterFloatArray->value = parameter.af;
                    runtime_operator->params.insert({name, runtimeParameterFloatArray});
                }
                    break;
                case int(RuntimeParameterType::kParameterStringArray): {
                    RuntimeParameterStringArray *runtimeParameterStringArray = new RuntimeParameterStringArray();
                    runtimeParameterStringArray->value = parameter.as;
                    runtime_operator->params.insert({name, runtimeParameterStringArray});
                }
                    break;
                default: {
                    LOG(FATAL) << "Unknown parameter type";
                }
            }
        }
    }

    void RuntimeGraph::initGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                      const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &pair: attrs) {
            const std::string &name = pair.first;
            const pnnx::Attribute &attr = pair.second;
            switch (attr.type) {
                case 1: {
                    std::shared_ptr<RuntimeAttribute> runtimeAttribute = std::make_shared<RuntimeAttribute>();
                    runtimeAttribute->type = RuntimeDataType::KTypeFloat32;
                    runtimeAttribute->weight_data = attr.data;
                    runtimeAttribute->shape = attr.shape;
                    runtime_operator->attribute.insert({name, runtimeAttribute});
                }
                    break;
                default: {
                    LOG(FATAL) << "Unknown attribute type";
                }
            }
        }
    }

    const std::vector<std::shared_ptr<RuntimeOperator>> RuntimeGraph::operators() const {
        return this->operators_;
    }
}