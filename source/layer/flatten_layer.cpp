//
// Created by zpx on 2023/02/26.
//
#include "layer/flatten_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    FlattenLayer::FlattenLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("Flatten") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorFlatten)
                        << "The operator is illegal: " << int(op->op_type_);
        FlattenOperator *flattenOperator = dynamic_cast<FlattenOperator *>(op.get());
        this->op_ = std::make_unique<FlattenOperator>(flattenOperator->getStartDim(), flattenOperator->getEndDim());
    }

    void FlattenLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(op_ != nullptr && op_->op_type_ == OpType::kOperatorFlatten);
        CHECK(!inputs.empty()) << "The input feature map is empty";
        CHECK(op_->getStartDim() >= 0 && op_->getEndDim() <= 3) << "The start dim or end dim  is not correct";
        const uint32_t batch_size = inputs.size();
        uint8_t new_dim = 1;
        std::vector<uint32_t> current_shape = inputs[0]->shapes(); // 旧张量的形状
        const uint32_t start_dim = op_->getStartDim();
        const uint32_t end_dim = op_->getEndDim() == -1 ? current_shape.size() : op_->getEndDim();
        std::vector<uint32_t> new_shape;
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        for (int j = 0; j < current_shape.size(); ++j) { // 获取新的张量形状
            if (j < start_dim || j > end_dim) {
                new_shape.push_back(current_shape[j]);
            } else {
                new_dim *= current_shape[j];
                if (j == end_dim)new_shape.push_back(new_dim);
            }
        }
#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; ++i) {
            auto input_data = inputs.at(i)->clone();
            input_data->reRawView(new_shape);           // 改变张量形状
            outputs[i] = input_data;
        }
    }

    std::shared_ptr<Layer> FlattenLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> flatterLayer = std::make_shared<FlattenLayer>(op);
        return flatterLayer;
    }

    void FlattenLayer::Forwards() {
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

    LayerRegistererWrapper flattenRegisterWrapper(OpType::kOperatorFlatten, FlattenLayer::CreateInstance);
}