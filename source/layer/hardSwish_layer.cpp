//
// Created by zpx on 2023/02/26.
//
#include "layer/hardSwish_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    HardSwishLayer::HardSwishLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("HardSwish") {
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
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto input_data = inputs.at(i)->clone();
            CHECK(input_data != nullptr && !input_data->empty()) << "Input data of hardSigmoid is illegal";
            input_data->data().transform([&](float value) {
                float tmp = value + 3.0f;
                if (tmp < 0)tmp = 0.f;
                if (tmp > 6)tmp = 6.f;
                return value * (tmp / 6.0f);
            });
            outputs[i] = input_data;
        }
    }

    std::shared_ptr<Layer> HardSwishLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> hardSwishLayer = std::make_shared<HardSwishLayer>(op);
        return hardSwishLayer;
    }

    void HardSwishLayer::Forwards() {
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

    LayerRegistererWrapper kHardSwishLayer(OpType::kOperatorHardSwish, HardSwishLayer::CreateInstance);
}