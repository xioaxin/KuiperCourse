//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include "layer/avgPooling_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    AvgPoolingLayer::AvgPoolingLayer(const std::shared_ptr<Operator> &op) : Layer("avgPooling") {
        CHECK(op->op_type_ == OpType::kOperatorAvgPooling) << "Operator has a wrong type: " << int(op->op_type_);
        AvgPoolingOperator *avgPoolingOperator = dynamic_cast<AvgPoolingOperator *>(op.get());
        CHECK(avgPoolingOperator != nullptr) << "AvgPooling operator is empty";
        this->op_ = std::make_unique<AvgPoolingOperator>(*avgPoolingOperator);
    }

    void AvgPoolingLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorAvgPooling);
        CHECK(!inputs.empty());
        const uint32_t padding_h = this->op_->get_padding_h();
        const uint32_t padding_w = this->op_->get_padding_w();
        const uint32_t kernel_h = this->op_->get_pooling_h();
        const uint32_t kernel_w = this->op_->get_pooling_w();
        const uint32_t stride_h = this->op_->get_stride_h();
        const uint32_t stride_w = this->op_->get_stride_w();
        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; i++) {
            const std::shared_ptr<Tensor<float>> &input_data_ = inputs.at(i);
            input_data_->padding({padding_h, padding_h, padding_w, padding_w}, std::numeric_limits<float>::lowest());
            const uint32_t input_h = input_data_->rows();
            const uint32_t input_w = input_data_->cols();
            const uint32_t input_c = input_data_->channels();
            const uint32_t output_c = input_c;
            const uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
            const uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
            CHECK(output_w > 0 && output_h > 0);
            std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(output_c, output_h, output_w);
            for (uint32_t ic = 0; ic < input_c; ic++) {
                const arma::fmat &input_channel = input_data_->at(ic);
                arma::fmat &output_channel = output_data->at(ic);
                for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
                    for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
                        const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
                        // TODO
                        output_channel.at(int(r / stride_h), int(c / stride_w)) = arma::mean(arma::mean(region));
                    }
                }
            }
            outputs.push_back(output_data);
        }
    }

    std::shared_ptr<Layer> AvgPoolingLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        CHECK(op->op_type_ == OpType::kOperatorAvgPooling);
        std::shared_ptr<Layer> max_layer = std::make_shared<AvgPoolingLayer>(op);
        return max_layer;
    }

    LayerRegistererWrapper kAvgPoolingLayer(OpType::kOperatorAvgPooling, AvgPoolingLayer::CreateInstance);
}