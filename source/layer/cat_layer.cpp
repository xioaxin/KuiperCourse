//
// Created by zpx on 2023/02/25.
//
#include "layer/cat_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    CatLayer::CatLayer(const std::shared_ptr<Operator> &op) : Layer("Cat") {
        CHECK(op != nullptr && op->op_type_ == OpType::kOperatorCat)
                        << "The operator is wrong of " << int(op->op_type_);
        CatOperator *catOperator = dynamic_cast<CatOperator *>(op.get());
        CHECK(catOperator != nullptr) << "Cat operator is empty";
        this->op_ = std::make_unique<CatOperator>(catOperator->getDim());
    }

    void CatLayer::Forwards(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs) {
        CHECK(this->op_ && this->op_->op_type_ == OpType::kOperatorCat);
        CHECK(!inputs.empty()) << "The input of cat layer is empty";
        if (inputs.size() == outputs.size()) {
            LOG(FATAL) << "The input and output size is not adapting";
        }
        const uint32_t dim = this->op_->getDim();
        if (dim != 1 && dim != 3) {
            LOG(FATAL) << "The dimension of cat layer is error";
        }
        const uint32_t output_size = outputs.size();
        CHECK(inputs.size() % output_size == 0);
        const uint32_t packet_size = inputs.size() / output_size;
        for (uint32_t i = 0; i < output_size; ++i) {
            std::shared_ptr<ftensor> output = outputs.at(i);
            uint32_t start_channel = 0;
            uint32_t rows = inputs.front()->rows();
            uint32_t cols = inputs.front()->cols();
            for (int j = i; j < inputs.size(); ++j) {
                const std::shared_ptr<ftensor> &input = inputs.at(j);
                CHECK(input != nullptr && !input->empty()) << "The input feature map of cat layer is empty";
                const uint32_t in_channels = input->channels();
                CHECK(rows == input->rows() && cols == input->cols());
                if (output == nullptr || output->empty()) {
                    output = std::make_shared<ftensor>(in_channels * packet_size, rows, cols);
                    outputs.at(i) = output;
                }
                CHECK(output->channels() == in_channels * packet_size && output->rows() == rows &&
                      output->cols() == cols);
                for(uint32_t c=0;c<in_channels;++c){
                    output->at(start_channel+c)=input->at(c);
                }
                start_channel += input->channels();
            }
        }
    }

    std::shared_ptr<Layer> CatLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> catLayer = std::make_shared<CatLayer>(op);
        return catLayer;
    }

    LayerRegistererWrapper kCatLayer(OpType::kOperatorCat, CatLayer::CreateInstance);
}
