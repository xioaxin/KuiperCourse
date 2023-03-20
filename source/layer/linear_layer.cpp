//
// Created by zpx on 2023/02/28.
//
#include "layer/linear_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    LinearLayer::LinearLayer(const std::shared_ptr<Operator> &op) : Layer("Linear") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorLinear)
                        << "The operator of linear layer is illegal: " << int(op->op_type_);
        LinearOperator *linearOperator = dynamic_cast<LinearOperator *>(op.get());
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
        const std::shared_ptr<ftensor> weight = this->op_->getWeights();
        const std::shared_ptr<ftensor> bias = this->op_->getBias();
        const uint32_t input_feature = this->op_->getInputFeature();
        const uint32_t output_feature = this->op_->getOutputFeature();
        CHECK(weight != nullptr && !weight->empty()) << "The weight of linear layer is empty";
        outputs.clear();
        for (uint32_t i = 0; i < batch_size; i++) {
            auto input_data = inputs.at(i)->clone();
            CHECK(input_data->channels() == 1) << "The channel of input data must equal to 1";
            CHECK(input_data != nullptr && !input_data->empty()) << "The input data of linear layer is empty";
            CHECK(weight->rows() == input_data->cols() && weight->cols() == output_feature)
                            << "The shape of input data and weight is illegal";
            CHECK(input_data->cols() == input_feature)
                            << "The input feature dim setting is not equal to the input tensor dim";
            std::shared_ptr<ftensor> output_data = std::make_shared<ftensor>(1, input_data->rows(), output_feature);
            arma::fmat A = input_data->data();
            arma::fmat B = weight->data();
            if (use_bias) {
                CHECK(bias->cols() == output_feature) << "The rows of bias is not equal to output feature dim";
                arma::fmat C = bias->data();
                output_data->at(0) = A * B + C;
            } else {
                output_data->at(0) = A * B;
            }
            outputs.push_back(output_data);
        }
    }

    std::shared_ptr<Layer> LinearLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> linearLayer = std::make_shared<LinearLayer>(op);
        return linearLayer;
    }

    LayerRegistererWrapper linearLayerRegistererWrapper(OpType::kOperatorLinear, LinearLayer::CreateInstance);
}