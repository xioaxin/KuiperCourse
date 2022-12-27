//
// Created by zpx on 2022/12/26.
//
#include <glog/logging.h>
#include "ops/relu_op.h"
#include "layer/layer.h"
#include "layer/relu_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
        CHECK(op->op_type_ == OpType::kOperatorRelu) << "Operator was a wrong typr: " << int(op->op_type_);
        ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());
        CHECK(relu_op != nullptr) << "Relu operator is empty";
        this->op_ = std::make_unique<ReluOperator>(relu_op->get_thresh());
    }

    void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                             std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorRelu);
        const uint32_t batch_size = inputs.size();
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
            input_data->data().transform([&](float value) {
                float thresh = op_->get_thresh();
                if (value <= thresh) { return 0.f; }
                else {
                    return value;
                }
            });
            outputs.push_back(input_data);
        }
    }

    std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
        return relu_layer;
    }

    LayerRegistererWrapper kReluLayer(OpType::kOperatorRelu, ReluLayer::CreateInstance);
}