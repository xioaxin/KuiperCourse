//
// Created by zpx on 2023/02/28.
//
#include "layer/linear_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    LinearLayer::LinearLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("Linear") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorLinear)
                        << "The operator of linear layer is illegal: " << int(op->op_type_);
        auto *linearOperator = dynamic_cast<LinearOperator *>(op.get());
        CHECK(linearOperator != nullptr) << "The operator of linear layer is empty";
        this->op_ = std::make_unique<LinearOperator>(*linearOperator);
    }

    void LinearLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorLinear)
                        << "The operator of linear layer is illegal";
        CHECK(!inputs.empty()) << "The input of linear layer is empty";
        const uint32_t batch_size = inputs.size();
        const bool use_bias = this->op_->getUseBias();
        const std::vector<sftensor> weight = this->op_->getWeight();
        const std::vector<sftensor> bias = this->op_->getBias();
        const uint32_t input_feature = this->op_->getInputFeature();
        const uint32_t output_feature = this->op_->getOutputFeature();
        CHECK(!weight.empty()) << "The weight of linear layer is empty";
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        if (inputs[0]->cols() == 1) {
//#pragma omp parallel num_threads(batch_size)
            for (uint32_t i = 0; i < batch_size; i++) {
                auto &input_data = inputs.at(i);
                CHECK(input_data->channels() == 1) << "The channel of input data must equal to 1";
                CHECK(input_data != nullptr && !input_data->empty()) << "The input data of linear layer is empty";
                CHECK(weight[i]->cols() == input_data->rows() && weight[i]->rows() == output_feature)
                                << "The shape of input data and weight is illegal";
                std::shared_ptr<ftensor> output_data = std::make_shared<ftensor>(1, output_feature, 1);
                arma::fmat A = input_data->data();
                arma::fmat B = weight[i]->data();
                arma::fmat tmp = B * A;
                if (use_bias) {
                    CHECK(bias[i]->rows() == output_feature) << "The rows of bias is not equal to output feature dim";
                    arma::fmat C = bias[i]->data();
                    output_data->at(0) = tmp + C;
                } else {
                    output_data->at(0) = tmp;
                }
                outputs[i] = output_data;
            }
        } else {
//#pragma omp parallel num_threads(batch_size)
            for (uint32_t i = 0; i < batch_size; i++) {
                auto &input_data = inputs.at(i);
                CHECK(input_data->channels() == 1) << "The channel of input data must equal to 1";
                CHECK(input_data != nullptr && !input_data->empty()) << "The input data of linear layer is empty";
                std::shared_ptr<ftensor> output_data = std::make_shared<ftensor>(1, input_data->rows(), output_feature);
                arma::fmat A = input_data->data();
                arma::fmat B = weight[i]->data();
                arma::fmat tmp = A * B.t();
                if (use_bias) {
                    CHECK(bias[i]->cols() == output_feature) << "The rows of bias is not equal to output feature dim";
                    arma::fmat C = bias[i]->data();
                    output_data->at(0) = tmp + C;
                } else {
                    output_data->at(0) = tmp;
                }
                outputs[i] = output_data;
            }
        }
    }

    std::shared_ptr<Layer> LinearLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> linearLayer = std::make_shared<LinearLayer>(op);
        return linearLayer;
    }

    void LinearLayer::Forwards() {
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

    LayerRegistererWrapper linearLayerRegistererWrapper(OpType::kOperatorLinear, LinearLayer::CreateInstance);
}