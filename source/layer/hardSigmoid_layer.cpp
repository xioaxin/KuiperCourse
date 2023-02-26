// Reference: https://paperswithcode.com/method/hard-sigmoid
// Created by zpx on 2023/02/26.
//
#include "layer/hardSigmoid_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    HardSigmoidLayer::HardSigmoidLayer(const std::shared_ptr<Operator> &op) : Layer("HardSigmoid") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorHardSigmoid)
                        << "The operation is illegal: " << int(op->op_type_);
        HardSigmoidOperator *hardSigmoidOperator = dynamic_cast<HardSigmoidOperator *>(op.get());
        CHECK(hardSigmoidOperator != nullptr);
        this->op_ = std::make_unique<HardSigmoidOperator>(*hardSigmoidOperator);
    }

    void HardSigmoidLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(op_ != nullptr && op_->op_type_ == OpType::kOperatorHardSigmoid)
                        << "The operator of hardSigmoidLayer is illegal";
        CHECK(!inputs.empty()) << "The input of hardSigmoidLayer is empty";
        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            auto &input_data = inputs.at(i);
            CHECK(input_data != nullptr && !input_data->empty()) << "Input data of hardSigmoid is illegal";
            input_data->data().transform([&](float value) {
                value = (value + 1) / 2;
                if (value > 1)value = 1.0f;
                if (value < 0)value = 0.f;
                return value;
            });
            outputs.push_back(input_data);
        }
    }

    std::shared_ptr<Layer> HardSigmoidLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> hardSigmoidLayer = std::make_shared<HardSigmoidLayer>(op);
        return hardSigmoidLayer;
    }

    LayerRegistererWrapper hardSigmoidLayerRegisterWrapper(OpType::kOperatorHardSigmoid,
                                                           HardSigmoidLayer::CreateInstance);
}