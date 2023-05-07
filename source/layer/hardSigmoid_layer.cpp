// Reference: https://paperswithcode.com/method/hard-sigmoid
// Created by zpx on 2023/02/26.
//
#include "layer/hardSigmoid_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    HardSigmoidLayer::HardSigmoidLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("HardSigmoid") {
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
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto input_data = inputs.at(i)->clone();
            CHECK(input_data != nullptr && !input_data->empty()) << "Input data of hardSigmoid is illegal";
            input_data->data().transform([](float value) {
                value = (value + 1) / 2.0f;
                if (value > 1)value = 1.0f;
                if (value < 0)value = 0.f;
                return value;
            });
            outputs[i] = input_data;
        }
    }

    std::shared_ptr<Layer> HardSigmoidLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> hardSigmoidLayer = std::make_shared<HardSigmoidLayer>(op);
        return hardSigmoidLayer;
    }

    void HardSigmoidLayer::Forwards() {
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

    LayerRegistererWrapper hardSigmoidLayerRegisterWrapper(OpType::kOperatorHardSigmoid,
                                                           HardSigmoidLayer::CreateInstance);
}