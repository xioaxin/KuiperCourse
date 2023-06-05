//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include "layer/adaptiveMaxPooling_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    AdaptiveMaxPoolingLayer::AdaptiveMaxPoolingLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer(
            "AdaptiveMaxPooling") {
        CHECK(op->op_type_ == OpType::kOperatorAdaptiveMaxPooling)
                        << "Operator has a wrong type: " << int(op->op_type_);
        AdaptiveMaxPoolingOperator *adaptiveMaxPoolingOperator = dynamic_cast<AdaptiveMaxPoolingOperator *>(op.get());
        CHECK(adaptiveMaxPoolingOperator != nullptr) << "AdaptiveMaxPooling operator is empty";
        this->op_ = std::make_unique<AdaptiveMaxPoolingOperator>(*adaptiveMaxPoolingOperator);
    }

    void AdaptiveMaxPoolingLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                           std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorAdaptiveMaxPooling);
        CHECK(!inputs.empty());
        const uint32_t batch_size = inputs.size();
        const uint32_t input_channels = inputs[0]->channels();
        const uint32_t input_rows = inputs[0]->rows();
        const uint32_t input_cols = inputs[0]->cols();
        const uint32_t output_rows = this->op_->getOutputSize()[0];
        const uint32_t output_cols = this->op_->getOutputSize()[1];
        CHECK(output_rows > 0 && output_cols > 0);
        // 池化层参数
        const uint32_t stride_h = uint32_t(floor(input_rows / output_rows));
        const uint32_t stride_w = uint32_t(floor(input_cols / output_cols));
        const uint32_t kernel_h = input_rows - (output_rows - 1) * stride_h;
        const uint32_t kernel_w = input_cols - (output_cols - 1) * stride_w;
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; i++) {
            std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(input_channels, output_rows,
                                                                                         output_cols);
            const std::shared_ptr<Tensor<float>> &input_data_ = inputs.at(i)->clone();
            for (uint32_t ic = 0; ic < input_channels; ic++) {
                const arma::fmat &input_channel = input_data_->at(ic);
                arma::fmat &output_channel = output_data->at(ic);
                for (uint32_t r = 0; r < input_rows - kernel_h + 1; r += stride_h) {
                    for (uint32_t c = 0; c < input_cols - kernel_w + 1; c += stride_w) {
                        const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
                        output_channel.at(int(r / stride_h), int(c / stride_w)) = region.max();
                    }
                }
            }
            outputs[i] = output_data;
        }
    }

    std::shared_ptr<Layer> AdaptiveMaxPoolingLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        CHECK(op->op_type_ == OpType::kOperatorAdaptiveMaxPooling);
        std::shared_ptr<Layer> adaptiveMaxLayer = std::make_shared<AdaptiveMaxPoolingLayer>(op);
        return adaptiveMaxLayer;
    }

    void AdaptiveMaxPoolingLayer::Forwards() {
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

    LayerRegistererWrapper kAdaptiveMaxPoolingLayer(OpType::kOperatorAdaptiveMaxPooling, AdaptiveMaxPoolingLayer::CreateInstance);
}