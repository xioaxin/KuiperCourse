//
// Created by zpx on 2023/02/26.
//
#include "layer/hardSwish_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    HardSwishLayer::HardSwishLayer(const std::shared_ptr<Operator> &op) : Layer("HardSwish") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorHardSwish)
                        << "The operator is illegal for hardSwish layer: " << int(op->op_type_);
        HardSwishOperator *hardSwishOperator = dynamic_cast<HardSwishOperator *>(op.get());
        CHECK(hardSwishOperator != nullptr);
        this->op_ = std::make_unique<HardSwishOperator>(*hardSwishOperator);
    }

    void HardSwishLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(op_ != nullptr && op_->op_type_ == OpType::kOperatorHardSwish)
                        << "The operator of hardSwishLayer is illegal";
        CHECK(!inputs.empty()) << "The input of hardSwishLayer is empty";
        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto input_data = inputs.at(i)->clone();
            CHECK(input_data != nullptr && !input_data->empty()) << "Input data of hardSigmoid is illegal";
            input_data->data().transform([&](float value) {
                float tmp = value + 3.0f;
                if (tmp < 0)tmp = 0.f;
                if (tmp > 6)tmp = 6.f;
                return value * (tmp / 6.0f);
            });
            outputs.push_back(input_data);
        }
    }

    std::shared_ptr<Layer> HardSwishLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> hardSwishLayer = std::make_shared<HardSwishLayer>(op);
        return hardSwishLayer;
    }

    LayerRegistererWrapper kHardSwishLayer(OpType::kOperatorHardSwish, HardSwishLayer::CreateInstance);
}