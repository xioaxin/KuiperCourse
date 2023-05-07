//
// Created by zpx on 2023/01/01.
//
#include <glog/logging.h>
#include "ops/leakyRelu_op.h"
#include "layer/leakyRelu_layer.h"

namespace kuiper_infer {
    LeakyReluLayer::LeakyReluLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("LeakyRelu") {
        CHECK(op->op_type_ == OpType::kOperatorLeakyRelu) << "Operator was a wrong type: " << int(op->op_type_);
        LeakyReluOperator *leakyRelu_op = dynamic_cast<LeakyReluOperator *>(op.get());
        CHECK(leakyRelu_op != nullptr) << "Relu operator is empty";
        this->op_ = std::make_unique<LeakyReluOperator>(leakyRelu_op->get_thresh());
    }

    void LeakyReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorLeakyRelu);
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i)->clone();
            input_data->data().transform([&](float value) {
                float thresh = op_->get_thresh();
                if (value <= thresh) { return 0.f; }
                else {
                    return value;
                }
            });
            outputs[i]=input_data;
        }
    }

    std::shared_ptr<Layer> LeakyReluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> relu_layer = std::make_shared<LeakyReluLayer>(op);
        return relu_layer;
    }
    void LeakyReluLayer::Forwards() {
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
    LayerRegistererWrapper kLeakyReluLayer(OpType::kOperatorLeakyRelu, LeakyReluLayer::CreateInstance);
}