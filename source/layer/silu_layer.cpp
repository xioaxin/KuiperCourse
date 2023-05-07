//
// Created by zpx on 2023/02/28.
//
#include "layer/silu_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    SiluLayer::SiluLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("Silu") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorSilu)
                        << "The operator of siluLayer is illegal: " << int(op->op_type_);
        SiluOperator *siluOperator = dynamic_cast<SiluOperator *>(op.get());
        CHECK(siluOperator != nullptr) << "The operator of siluLayer is empty";
        this->op_ = std::make_unique<SiluOperator>(*siluOperator);
    }

    void SiluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                             std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(op_ != nullptr && op_->op_type_ == OpType::kOperatorSilu) << "The operation of siluLayer is illegal";
        CHECK(!inputs.empty()) << "The input data of siluLayer is empty";
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; i++) {
            const auto input_data = inputs.at(i)->clone();
            input_data->transform([&](float value) {
                value = value * 1 / float(1 + exp(-value));
                return value;
            });
            outputs[i]=input_data;
        }
    }

    std::shared_ptr<Layer> SiluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> siluLayer = std::make_shared<SiluLayer>(op);
        return siluLayer;
    }
    void SiluLayer::Forwards() {
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
    LayerRegistererWrapper siluLayerRegisterWrapper(OpType::kOperatorSilu, SiluLayer::CreateInstance);
}