//
// Created by zpx on 2023/01/01.
//
#include <glog/logging.h>
#include "ops/sigmoid_op.h"
#include "layer/layer.h"
#include "layer/sigmoid_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    SigmoidLayer::SigmoidLayer(const std::shared_ptr<Operator> &op) : Layer("Sigmoid") {
        CHECK(op->op_type_ == OpType::kOperatorSigmoid) << "Operator was a wrong type: " << int(op->op_type_);
        SigmoidOperator *sigmoid_op = dynamic_cast<SigmoidOperator *>(op.get());
        CHECK(sigmoid_op != nullptr) << "Sigmoid operator is empty";
        this->op_ = std::make_unique<SigmoidOperator>();
    }

    void SigmoidLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorSigmoid);
        const uint32_t batch_size = inputs.size();
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            const std::shared_ptr<Tensor<float>> input_data = inputs.at(i)->clone();
            input_data->data().transform([&](float value) {
                return 1 / (1 + std::exp(-value));
            });
            outputs.push_back(input_data);
        }
    }

    std::shared_ptr<Layer> SigmoidLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> sigmoid_layer = std::make_shared<SigmoidLayer>(op);
        return sigmoid_layer;
    }

    LayerRegistererWrapper kSigmoidLayer(OpType::kOperatorSigmoid, SigmoidLayer::CreateInstance);
}