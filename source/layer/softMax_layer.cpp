//
// Created by zpx on 2023/02/25.
//
#include "layer/softMax_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    SoftMaxLayer::SoftMaxLayer(const std::shared_ptr<Operator> &op) : Layer("SoftMax") {
        CHECK(op->op_type_ == OpType::kOperatorSoftMax) << "Operator was a wrong type: " << int(op->op_type_);
        SoftMaxOperator *softMaxOperator = dynamic_cast<SoftMaxOperator *>(op.get());
        CHECK(softMaxOperator != nullptr) << "Softmax operator is empty";
        this->op_ = std::make_unique<SoftMaxOperator>(*softMaxOperator);
    }

    void SoftMaxLayer::Forwards(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs) {
        CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorSoftMax);
        CHECK(!inputs.empty());
        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto &input_data = inputs.at(i);
            CHECK(input_data != nullptr && !input_data->empty()) << "The input feature map for softmax layer is empty";
            auto &output_data = outputs.at(i);
            if (output_data == nullptr || output_data->empty()) {
                output_data = std::make_shared<ftensor>(input_data->shapes());
                outputs.at(i) = output_data;
            }
            CHECK(input_data->shapes() == output_data->shapes()) << "The output size of softmax is error";
            const arma::fcube &input_data_ = input_data->data();
            const arma::fmat sum = arma::sum(arma::exp(input_data_), 2);
            for (uint32_t j = 0; j < input_data->channels(); ++j) {
                output_data->at(j) = arma::exp(input_data->at(j)) / sum;
            }
        }
    }

    std::shared_ptr<Layer> SoftMaxLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> softMax_layer = std::make_shared<SoftMaxLayer>(op);
        return softMax_layer;
    }

    LayerRegistererWrapper kSoftMaxLayer(OpType::kOperatorSoftMax, SoftMaxLayer::CreateInstance);
}