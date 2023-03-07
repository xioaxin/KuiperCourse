//
// Created by zpx on 2023/01/01.
//
#include <glog/logging.h>
#include "ops/leakyRelu_op.h"
#include "layer/layer.h"
#include "layer/leakyRelu_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    LeakyReluLayer::LeakyReluLayer(const std::shared_ptr<Operator> &op) : Layer("LeakyRelu") {
        CHECK(op->op_type_ == OpType::kOperatorLeakyRelu) << "Operator was a wrong type: " << int(op->op_type_);
        LeakyReluOperator *leakyRelu_op = dynamic_cast<LeakyReluOperator *>(op.get());
        CHECK(leakyRelu_op != nullptr) << "Relu operator is empty";
        this->op_ = std::make_unique<LeakyReluOperator>(leakyRelu_op->get_thresh());
    }

    void LeakyReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                             std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorLeakyRelu);
        const uint32_t batch_size = inputs.size();
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i)->clone();
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

    std::shared_ptr<Layer> LeakyReluLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> relu_layer = std::make_shared<LeakyReluLayer>(op);
        return relu_layer;
    }

    LayerRegistererWrapper kLeakyReluLayer(OpType::kOperatorLeakyRelu, LeakyReluLayer::CreateInstance);
}