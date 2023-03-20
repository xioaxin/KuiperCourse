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
#include "ops/cat_op.h"

namespace kuiper_infer {
    void RuntimeGraphShape::initOperatorInputTensor(
            const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
        if (operators.empty()) {
            LOG(ERROR) << "Operators for init input shapes is empty!";
            return;
        }
        for (const auto &op: operators) {
            if (op->input_operands.empty()) {
                continue;
            } else {
                const std::map<std::string, std::shared_ptr<RuntimeOperand>> &input_operands_map = op->input_operands;
                for (const auto &input_operand_iter: input_operands_map) {
                    const auto &input_operand = input_operand_iter.second;
                    const auto &type = input_operand->type;
                    CHECK(type == RuntimeDataType::KTypeFloat32) << "The graph only support float32 yet!";
                    const auto &input_operand_shape = input_operand->shapes;
                    auto &input_datas = input_operand->datas;
                    CHECK(!input_operand_shape.empty());
                    const int32_t batch = input_operand_shape.at(0);
                    CHECK(batch >= 0) << "Dynamic batch size is not supported!";
                    CHECK(input_operand_shape.size() == 2 || input_operand_shape.size() == 4 ||
                          input_operand_shape.size() == 3) << "Unsupported tensor shape sizes: "
                                                           << input_operand_shape.size();
                    if (!input_datas.empty()) {
                        CHECK(input_datas.size() == batch) << "Batch size is wrong!";
                        for (int32_t i = 0; i < batch; ++i) {
                            const std::vector<uint32_t> &input_data_shape = input_datas.at(i)->shapes();
                            CHECK(input_data_shape.size() == 3)
                                            << "THe origin shape size of operator input data do not equals "
                                               "to three";
                            if (input_operand_shape.size() == 4) {
                                CHECK(input_data_shape.at(0) == input_operand_shape.at(1) &&
                                      input_data_shape.at(1) == input_operand_shape.at(2) &&
                                      input_data_shape.at(2) == input_operand_shape.at(3));
                            } else if (input_operand_shape.size() == 2) {
                                CHECK(input_data_shape.at(1) == input_operand_shape.at(1) &&
                                      input_data_shape.at(0) == 1 && input_data_shape.at(2) == 1);
                            } else {
                                // current shape size = 3
                                CHECK(input_data_shape.at(1) == input_operand_shape.at(1) &&
                                      input_data_shape.at(0) == 1 &&
                                      input_data_shape.at(2) == input_operand_shape.at(2));
                            }
                        }
                    } else {
                        input_datas.resize(batch);
                        for (int32_t i = 0; i < batch; ++i) {
                            if (input_operand_shape.size() == 4) {
                                input_datas.at(i) = std::make_shared<Tensor<float>>(
                                        input_operand_shape.at(1), input_operand_shape.at(2),
                                        input_operand_shape.at(3));
                            } else if (input_operand_shape.size() == 2) {
                                input_datas.at(i) = std::make_shared<Tensor<float>>(
                                        1, input_operand_shape.at(1), 1);
                            } else {
                                // current shape is 3
                                input_datas.at(i) = std::make_shared<Tensor<float>>(1, input_operand_shape.at(1),
                                                                                    input_operand_shape.at(2));
                            }
                        }
                    }
                }
            }
        }
    }

    void RuntimeGraph::build(const std::string &input_name, const std::string &output_name) {
        if (graphState_ == GraphState::NeedInit) {
            bool init_graph = init();
            LOG_IF(FATAL, !init_graph) << "Init graph failed!";
        }
        CHECK(graphState_ >= GraphState::NeedBuild)
                        << "Graph status error, current state is " << int(graphState_);
        LOG_IF(FATAL, this->operators_.empty())
                        << "Graph operators is empty, may be no init";
        this->input_operators_maps_.clear();
        this->output_operators_maps_.clear();
        for (const auto &kOperator: this->operators_) {
            if (kOperator->type == "pnnx.Input") {
                this->input_operators_maps_.insert({kOperator->name, kOperator});
            } else if (kOperator->type == "pnnx.Output") {
                this->output_operators_maps_.insert({kOperator->name, kOperator});
            } else {
                // TODO: 以后的课中加layer的
                kOperator->layer=LayerRegisterer::CreateLayer(std::make_shared<CatOperator>(1));
            }
        }
        RuntimeGraphShape::initOperatorInputTensor(operators_);
        RuntimeGraphShape::initOperatorOutputTensor(graph_->ops, operators_);
        graphState_ = GraphState::Complete;
        input_name_ = input_name;
        output_name_ = output_name;
    }

    void RuntimeGraphShape::initOperatorOutputTensor(const std::vector<pnnx::Operator *> &pnnx_operators,
                                                     const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
        CHECK(!pnnx_operators.empty() && !operators.empty());
        CHECK(pnnx_operators.size() == operators.size());
        for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
            const std::vector<pnnx::Operand *> operands = pnnx_operators.at(i)->outputs;
            CHECK(operands.size() <= 1) << "Only support one node one output yet!";
            if (operands.empty()) {
                continue;
            }
            CHECK(operands.size() == 1) << "Only support one output in the KuiperInfer";
            pnnx::Operand *operand = operands.front();
            const auto &runtime_op = operators.at(i);
            CHECK(operand != nullptr) << "Operand output is null";
            const std::vector<int32_t> &operand_shapes = operand->shape;
            const auto &output_tensors = runtime_op->output_operands;
            const int32_t batch = operand_shapes.at(0);
            CHECK(batch >= 0) << "Dynamic batch size is not supported!";
            CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 4 ||
                  operand_shapes.size() == 3)
                            << "Unsupported shape sizes: " << operand_shapes.size();
            if (!output_tensors) {
                std::shared_ptr<RuntimeOperand> output_operand =
                        std::make_shared<RuntimeOperand>();
                output_operand->shapes = operand_shapes;
                output_operand->type = RuntimeDataType::KTypeFloat32;
                output_operand->name = operand->name + "_output";
                for (int j = 0; j < batch; ++j) {
                    if (operand_shapes.size() == 4) {
                        output_operand->datas.push_back(std::make_shared<Tensor<float>>(
                                operand_shapes.at(1), operand_shapes.at(2),
                                operand_shapes.at(3)));
                    } else if (operand_shapes.size() == 2) {
                        output_operand->datas.push_back(
                                std::make_shared<Tensor<float>>(1, operand_shapes.at(1), 1));
                    } else {
                        // current shape is 3
                        output_operand->datas.push_back(std::make_shared<Tensor<float>>(
                                1, operand_shapes.at(1), operand_shapes.at(2)));
                    }
                }
                runtime_op->output_operands = std::move(output_operand);
            } else {
                CHECK(batch == output_tensors->datas.size());
                // output_tensors empty
                CHECK(output_tensors->type == RuntimeDataType::KTypeFloat32);
                CHECK(output_tensors->shapes == operand_shapes);
                for (uint32_t b = 0; b < batch; ++b) {
                    const std::vector<uint32_t> &tensor_shapes =
                            output_tensors->datas.at(b)->shapes();
                    if (operand_shapes.size() == 4) {
                        if (tensor_shapes.at(0) != operand_shapes.at(1) ||
                            tensor_shapes.at(1) != operand_shapes.at(2) ||
                            tensor_shapes.at(2) != operand_shapes.at(3)) {
                            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes = std::vector<uint32_t>{(uint32_t) operand_shapes.at(1),
                                                                              (uint32_t) operand_shapes.at(2),
                                                                              (uint32_t) operand_shapes.at(3)};
                            output_tensors->datas.at(b)->reRawShape(target_shapes);
                        }
                    } else if (operand_shapes.size() == 2) {
                        if (tensor_shapes.at(0) != 1 ||
                            tensor_shapes.at(1) != operand_shapes.at(1) ||
                            tensor_shapes.at(2) != 1) {
                            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes = std::vector<uint32_t>{1, (uint32_t) operand_shapes.at(1), 1};
                            output_tensors->datas.at(b)->reRawShape(target_shapes);
                        }
                    } else {
                        // current shape is 3
                        if (tensor_shapes.at(0) != 1 ||
                            tensor_shapes.at(1) != operand_shapes.at(1) ||
                            tensor_shapes.at(2) != operand_shapes.at(2)) {
                            DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
                            const auto &target_shapes =
                                    std::vector<uint32_t>{1, (uint32_t) operand_shapes.at(1),
                                                          (uint32_t) operand_shapes.at(2)};
                            output_tensors->datas.at(b)->reRawShape(target_shapes);
                        }
                    }
                }
            }
        }
    }

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
        for (const auto &current_operator: this->operators_) {
            const std::vector<std::string> &output_names = current_operator->output_names;
            for (const auto &next_operator: this->operators_) {
                if (next_operator == current_operator) {
                    continue;
                }
                if (std::find(output_names.begin(), output_names.end(), next_operator->name) != output_names.end()) {
                    current_operator->output_operators.insert({next_operator->name, next_operator});
                }
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

    //TODO: 待理解
    std::vector<std::shared_ptr<ftensor>>
    RuntimeGraph::forward(const std::vector<std::shared_ptr<ftensor>> &inputs, bool debug) {
        if (graphState_ < GraphState::Complete) {
            LOG(FATAL) << "Graph need be build";
        }
        CHECK(graphState_ == GraphState::Complete) << "Graph status error, current status is " << int(graphState_);
        std::shared_ptr<RuntimeOperator> input_operator;
        if (input_operators_maps_.find(input_name_) == input_operators_maps_.end()) {
            LOG(FATAL) << "Can not find the input node: " << input_name_;
        } else {
            input_operator = input_operators_maps_[input_name_];
        }
        std::shared_ptr<RuntimeOperator> output_operator;
        if (output_operators_maps_.find(output_name_) == output_operators_maps_.end()) {
            LOG(FATAL) << "Can not find the output node: " << output_name_;
        } else {
            output_operator = output_operators_maps_[output_name_];
        }
        std::deque<std::shared_ptr<RuntimeOperator>> operator_queue;
        operator_queue.push_back(input_operator);
        std::map<std::string, double> run_duration_infos;
        while (!operator_queue.empty()) {
            std::shared_ptr<RuntimeOperator> current_operator = operator_queue.front();
            operator_queue.pop_front();
            if (!current_operator || current_operator == output_operator) {
                LOG(INFO) << "Model infer end";
                break;
            }
            if (current_operator == input_operator) {
                probeNextLayer(current_operator, operator_queue, inputs);
            } else {
                std::string current_operator_name = current_operator->name;
                if (!checkOperatorReady(current_operator)) {
                    if (operator_queue.empty()) {
                        LOG(FATAL) << "Current operator is not ready!";
                        break;
                    } else {
                        operator_queue.push_back(current_operator);
                    }
                }
                const std::vector<std::shared_ptr<RuntimeOperand>> &input_operand_datas = current_operator->input_operands_seq;
                std::vector<std::shared_ptr<ftensor >> layer_input_datas;
                for (const auto &input_operand_data: input_operand_datas) {
                    for (const auto &input_data: input_operand_data->datas) {
                        layer_input_datas.push_back(input_data);
                    }
                }
                CHECK(!layer_input_datas.empty()) << "Layer input data is empty";
                CHECK(current_operator->output_operands != nullptr && !current_operator->output_operands->datas.empty())
                                << "Layer output data is empty";
                const auto &start = std::chrono::steady_clock::now();
                probeNextLayer(current_operator, operator_queue, current_operator->output_operands->datas);
                if (debug) {
                    LOG(INFO) << "current operator: " << current_operator->name;
                }
            }
        }
        for (const auto &op: this->operators_) {
            op->meet_num = 0;
        }
        CHECK(output_operator->input_operands.size() == 1) << "The graph only support one path to the output node yet!";
        const auto &output_operator_operand = output_operator->input_operands.begin();
        const auto &output_operand = output_operator_operand->second;
        return output_operand->datas;
    }

 // TODO：加强理解
    void RuntimeGraph::probeNextLayer(const std::shared_ptr<RuntimeOperator> &current_operator,
                                      std::deque<std::shared_ptr<RuntimeOperator>> &operator_queue,
                                      std::vector<std::shared_ptr<ftensor>> layer_output_data) {
        const auto &next_operators = current_operator->output_operators;
        std::vector<std::vector<std::shared_ptr<ftensor>>> next_input_data_arr;
        for (const auto &next_operator: next_operators) {
            const auto &next_rt_operator = next_operator.second;
            const auto &next_input_operands = next_rt_operator->input_operands;
            if (next_input_operands.find(current_operator->name) != next_input_operands.end()) {
                std::vector<std::shared_ptr<ftensor>> next_input_datas = next_input_operands.at(
                        current_operator->name)->datas;
                next_input_data_arr.push_back(next_input_datas);
                next_rt_operator->meet_num += 1;
                if (std::find(operator_queue.begin(), operator_queue.end(), next_rt_operator) == operator_queue.end()) {
                    if (checkOperatorReady(next_rt_operator)) {
                        operator_queue.push_back(next_rt_operator);
                    }
                }
            }
        }
        setOperatorInputData(layer_output_data, next_input_data_arr);
    }

    bool RuntimeGraph::checkOperatorReady(const std::shared_ptr<RuntimeOperator> &op) {
        CHECK(op != nullptr);
        CHECK(op->meet_num <= op->input_operands.size());
        return op->meet_num == op->input_operands.size();
    }

    void RuntimeGraph::setOperatorInputData(std::vector<std::shared_ptr<ftensor>> &src,
                                            std::vector<std::vector<std::shared_ptr<ftensor>>> &dest) {
        CHECK(!src.empty() && !dest.empty()) << "Src or dest array is empty";
        for (uint32_t j = 0; j < src.size(); j++) {
            const auto &src_data = src.at(j)->data();
            for (uint32_t i = 0; i < dest.size(); i++) {
                dest.at(i).at(j)->set_data(src_data);
            }
        }
    }
}