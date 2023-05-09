//
// Created by zpx on 2023/05/07.
//
#include "layer/view_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    ViewLayer::ViewLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("View") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorView) << "The operator is illegal: " << int(op->op_type_);
        ViewOperator *viewOperator = dynamic_cast<ViewOperator *>(op.get());
        this->op_ = std::make_unique<ViewOperator>(viewOperator->getShape());
    }

    void ViewLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                             std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(op_ != nullptr && op_->op_type_ == OpType::kOperatorView);
        CHECK(!inputs.empty()) << "The input feature map is empty";
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        const auto data_shape = inputs.at(0)->data();
        const auto layer_shape = this->op_->getShape();
        int count = 0, index = 0, sum = 1;
        for (int i = 0; i < layer_shape.size(); ++i) {
            if (layer_shape[i] == -1) {
                count++;
                index = i;
            } else sum *= layer_shape[i];
        }
        CHECK(sum <= data_shape.size() && count <= 1) << "the layer shape is not equal to data shape";
        std::vector<uint32_t> new_shape(3, 1);
        if (layer_shape.size() == 3) {
            for (int i = 0; i < layer_shape.size(); i++)new_shape[i] = layer_shape[i];
            if (layer_shape[index] == -1) new_shape[index] = data_shape.size() / sum;
        } else if (layer_shape.size() == 2) {
            for (int i = 0; i < layer_shape.size(); i++)new_shape[i + 1] = layer_shape[i];
            if (layer_shape[index] == -1) new_shape[index + 1] = data_shape.size() / sum;
        } else {
            LOG(FATAL) << "Illegal shape of viewLayer";
        }

//#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0; i < batch_size; ++i) {
            auto &input_data = inputs.at(i);
            input_data->reRawView(new_shape);           // 改变张量形状
            outputs[i] = input_data;
        }
    }

    std::shared_ptr<Layer> ViewLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> viewLayer = std::make_shared<ViewLayer>(op);
        return viewLayer;
    }

    void ViewLayer::Forwards() {
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

    LayerRegistererWrapper viewRegisterWrapper(OpType::kOperatorView, ViewLayer::CreateInstance);
}