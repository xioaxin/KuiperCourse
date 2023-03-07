//
// Created by zpx on 2023/02/25.
//
#include <glog/logging.h>
#include "layer/batchNorm_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    BatchNormLayer::BatchNormLayer(const std::shared_ptr<Operator> &op) : Layer("BatchNorm") {
        CHECK(op->op_type_ == OpType::kOperatorBatchNorm) << "Operator was a wrong type: " << int(op->op_type_);
        BatchNormOperator *batchNormOperator = dynamic_cast<BatchNormOperator *>(op.get());
        CHECK(batchNormOperator != nullptr) << "BatchNorm operator is empty";
        this->op_ = std::make_unique<BatchNormOperator>(*batchNormOperator);
    }

    void BatchNormLayer::Forwards(const std::vector<std::shared_ptr<ftensor>> &inputs,
                                  std::vector<std::shared_ptr<ftensor>> &outputs) {
        CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorBatchNorm);
        CHECK(!inputs.empty());
        CHECK(inputs.size() == outputs.size());
        const auto &mean_value = this->op_->getMeanValue();
        const auto &var_value = this->op_->getVarValue();
        CHECK(mean_value->shapes() == var_value->shapes()) << "The shape of mean value and var value are not correct";
        auto &affine_alpha = this->op_->getAffineAlpha();
        auto &affine_beta = this->op_->getAffineBata();
        CHECK(affine_alpha.size() == affine_beta.size()) << "The size of affine mean and affine var are not correct";
        const uint32_t batch_size = inputs.size();
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (uint32_t i = 0; i < batch_size; ++i) {
            const auto &input_data = inputs.at(i)->clone();
            CHECK(input_data != nullptr && !input_data->empty()) << "The input data is null or empty";
            CHECK(input_data->channels() == mean_value->channels())
                            << "The channel of input data and mean value size are not equal";
            CHECK(input_data->channels() == affine_alpha.size())
                            << "The channel of input data and affine value size are not correct";
            const auto &output_data = outputs.at(i);
            CHECK(output_data != nullptr && !output_data->empty()) << "The output data is null or empty";
            CHECK(input_data->size() == output_data->size())
                            << "Input data size and the output data size are not equal";
            for (uint32_t j = 0; j < mean_value->channels(); ++j) {
                CHECK(mean_value->at(j).size() == 1 && var_value->at(j).size() == 1);
                const float mean_value_ = mean_value->at(j, 0, 0);   // 均值
                const float var_value_ = std::sqrt(var_value->at(j, 0, 0) + this->op_->getEps()); // 方差
                CHECK(input_data->channels() >= 1)
                                << "The channel of the input feature maps and mean value is not adaption";
                output_data->at(j) = ((input_data->at(j) - mean_value_) / var_value_ * affine_alpha.at(j) +
                                      affine_beta.at(j));
            }
        }
    }

    std::shared_ptr<Layer> BatchNormLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
        std::shared_ptr<Layer> batch_norm_layer = std::make_unique<BatchNormLayer>(op);
        return batch_norm_layer;
    }

    LayerRegistererWrapper batchNormLayer(OpType::kOperatorBatchNorm, BatchNormLayer::CreateInstance);
}