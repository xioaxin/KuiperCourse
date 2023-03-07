//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include "layer/adaptiveAvgPooling_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    AdaptiveAvgPoolingLayer::AdaptiveAvgPoolingLayer(const std::shared_ptr<Operator> &op) : Layer(
            "AdaptiveAvgPooling") {
        CHECK(op->op_type_ == OpType::kOperatorAdaptiveAvgPooling)
                        << "Operator has a wrong type: " << int(op->op_type_);
        AdaptiveAvgPoolingOperator *adaptiveAvgPoolingOperator = dynamic_cast<AdaptiveAvgPoolingOperator *>(op.get());
        CHECK(adaptiveAvgPoolingOperator != nullptr) << "AdaptiveAvgPooling operator is empty";
        this->op_ = std::make_unique<AdaptiveAvgPoolingOperator>(*adaptiveAvgPoolingOperator);
    }

    void AdaptiveAvgPoolingLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                           std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorAdaptiveAvgPooling);
        CHECK(!inputs.empty());
        const uint32_t batch_size = inputs.size();
        const uint32_t input_channels = inputs[0]->channels();
        const uint32_t input_rows = inputs[0]->rows();
        const uint32_t input_cols = inputs[0]->cols();
        const uint32_t output_rows = this->op_->get_output_h();
        const uint32_t output_cols = this->op_->get_output_w();
        CHECK(output_rows > 0 && output_cols > 0);
        // 池化层参数
        const uint32_t stride_h = uint32_t(floor(input_rows / output_rows));
        const uint32_t stride_w = uint32_t(floor(input_cols / output_cols));
        const uint32_t kernel_h = input_rows - (output_rows - 1) * stride_h;
        const uint32_t kernel_w = input_cols - (output_cols - 1) * stride_w;
#ifdef OPENMP
#pragma omp parallel for
#endif
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
                        output_channel.at(int(r / stride_h), int(c / stride_w)) = arma::mean(arma::mean(region));
                    }
                }
            }
#ifdef OPENMP
#pragma omp critical
#endif
            outputs.push_back(output_data);
        }
    }

    std::shared_ptr<Layer> AdaptiveAvgPoolingLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        CHECK(op->op_type_ == OpType::kOperatorAdaptiveAvgPooling);
        std::shared_ptr<Layer> adaptiveAvgLayer = std::make_shared<AdaptiveAvgPoolingLayer>(op);
        return adaptiveAvgLayer;
    }

    LayerRegistererWrapper kAdaptiveAvgPoolingLayer(OpType::kOperatorAdaptiveAvgPooling,
                                                    AdaptiveAvgPoolingLayer::CreateInstance);
}