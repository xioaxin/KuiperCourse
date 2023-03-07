//
// Created by zpx on 2023/01/01.
//
#include <glog/logging.h>
#include "layer/relu6_layer.h"
#include "ops/relu6_op.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    Relu6Layer::Relu6Layer(const std::shared_ptr<Operator> &op) : Layer("Relu6") {
        CHECK(op->op_type_ == OpType::kOperatorRelu6) << "Operator was a wrong type: " << int(op->op_type_);
        Relu6Operator *relu_op = dynamic_cast<Relu6Operator *>(op.get());
        CHECK(relu_op != nullptr) << "Relu operator is empty";
        this->op_ = std::make_unique<Relu6Operator>(relu_op->get_thresh());
    }

    void Relu6Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorRelu6);
        const uint32_t batch_size = inputs.size();
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i)->clone();
            input_data->data().transform([&](float value) {
                return std::min(std::max(value, 0.f), op_->get_thresh());
            });
#ifdef OPENMP
#pragma omp critical
#endif
            outputs.push_back(input_data);
        }
    }

    std::shared_ptr<Layer> Relu6Layer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> relu6_layer = std::make_shared<Relu6Layer>(op);
        return relu6_layer;
    }

    LayerRegistererWrapper kRelu6Layer(OpType::kOperatorRelu6, Relu6Layer::CreateInstance);
}