//
// Created by zpx on 2023/01/01.
//
#include <glog/logging.h>
#include "layer/relu6_layer.h"
#include "ops/relu6_op.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    Relu6Layer::Relu6Layer(const std::shared_ptr<RuntimeOperator> &op) : Layer("Relu6") {
        CHECK(op->op_type_ == OpType::kOperatorRelu6) << "Operator was a wrong type: " << int(op->op_type_);
        Relu6Operator *relu_op = dynamic_cast<Relu6Operator *>(op.get());
        CHECK(relu_op != nullptr) << "Relu operator is empty";
        this->op_ = std::make_unique<Relu6Operator>(relu_op->get_thresh());
    }

    void Relu6Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorRelu6);
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i)->clone();
            input_data->data().transform([&](float value) {
                return std::min(std::max(value, 0.f), op_->get_thresh());
            });
            outputs[i]=input_data;
        }
    }

    std::shared_ptr<Layer> Relu6Layer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> relu6_layer = std::make_shared<Relu6Layer>(op);
        return relu6_layer;
    }
    void Relu6Layer::Forwards() {
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
    LayerRegistererWrapper kRelu6Layer(OpType::kOperatorRelu6, Relu6Layer::CreateInstance);
}