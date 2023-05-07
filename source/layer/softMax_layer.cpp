//
// Created by zpx on 2023/02/25.
//
#include "layer/softMax_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    SoftMaxLayer::SoftMaxLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("SoftMax") {
        CHECK(op->op_type_ == OpType::kOperatorSoftMax) << "Operator was a wrong type: " << int(op->op_type_);
        SoftMaxOperator *softMaxOperator = dynamic_cast<SoftMaxOperator *>(op.get());
        CHECK(softMaxOperator != nullptr) << "Softmax operator is empty";
        this->op_ = std::make_unique<SoftMaxOperator>(*softMaxOperator);
    }

    void SoftMaxLayer::Forwards(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs) {
        CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorSoftMax);
        CHECK(!inputs.empty());
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        for (uint32_t i = 0; i < batch_size; ++i) {
            auto &input_data = inputs.at(i);
            const auto channels = input_data->channels();
            CHECK(input_data != nullptr && !input_data->empty()) << "The input feature map for softmax layer is empty";
            const arma::fmat max_value = arma::max(input_data->data(), 2);
            arma::fmat sum_value(max_value.n_rows, max_value.n_cols, arma::fill::zeros);
            for (int j = 0; j < channels; ++j) {
                sum_value += arma::exp(input_data->at(j) - max_value);
            }
            for (int j = 0; j < channels; ++j) {
                input_data->at(j) = arma::exp(input_data->at(j) - max_value) / sum_value;
            }
            outputs[i] = input_data;
        }
    }

    std::shared_ptr<Layer> SoftMaxLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> softMax_layer = std::make_shared<SoftMaxLayer>(op);
        return softMax_layer;
    }

    void SoftMaxLayer::Forwards() {
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

    LayerRegistererWrapper kSoftMaxLayer(OpType::kOperatorSoftMax, SoftMaxLayer::CreateInstance);
}