//
// Created by zpx on 2023/03/01.
//
#include "layer/upSample_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    UpSampleLayer::UpSampleLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("UpSample") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorUpSample)
                        << "The operator of upSample is illegal: " << int(op->op_type_);
        UpSampleOperator *upSampleOperator = dynamic_cast<UpSampleOperator *>(op.get());
        CHECK(upSampleOperator != nullptr) << "The operator of upSample is empty";
        this->op_ = std::make_unique<UpSampleOperator>(*upSampleOperator);
    }

    void UpSampleLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(op_ != nullptr && op_->op_type_ == OpType::kOperatorUpSample)
                        << "The Operator of upSample Layer is illegal";
        CHECK(!inputs.empty()) << "The input of upSample layer is empty";
        const uint32_t batch_size = inputs.size();
        const float scale_h = this->op_->getScaleH();
        const float scale_w = this->op_->getScaleW();
        const uint32_t output_h = std::ceil(inputs[0]->rows() * scale_h);
        const uint32_t output_w = std::ceil(inputs[0]->cols() * scale_w);
        const UpSampleMode upSampleMode = this->op_->getUpSampleModel();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; ++i) {
            auto input_data = inputs[i]->clone();
            std::shared_ptr<ftensor> output_data = std::make_shared<ftensor>(input_data->channels(), output_h,output_w);
            for (uint32_t j = 0; j < input_data->channels(); j++) {
                arma::mat input_data_ = arma::conv_to<arma::mat>::from(input_data->at(j));
                arma::vec W = arma::linspace(0, input_data_.n_cols, input_data_.n_cols);
                arma::vec H = arma::linspace(0, input_data_.n_rows, input_data_.n_rows);
                arma::vec new_H = arma::linspace(H.min(), H.max(), output_h);
                arma::vec new_W = arma::linspace(W.min(), W.max(), output_w);
                arma::mat output_data_;
                switch (upSampleMode) {
                    case UpSampleMode::kModelNearest:
                        arma::interp2(W, H, input_data_, new_W, new_H, output_data_, "nearest");
                        break;
                    case UpSampleMode::kModelLinear:
                        arma::interp2(W, H, input_data_, new_W, new_H, output_data_, "linear");
                        break;
                }
                output_data->at(j) = arma::conv_to<arma::fmat>::from(output_data_);
            }
            outputs[i] = output_data;
        }
    }

    std::shared_ptr<Layer> UpSampleLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> upSampleLayer = std::make_shared<UpSampleLayer>(op);
        return upSampleLayer;
    }

    void UpSampleLayer::Forwards() {
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

    LayerRegistererWrapper upSampleLayerRegistererWrapper(OpType::kOperatorUpSample, UpSampleLayer::CreateInstance);
}