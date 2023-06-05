//
// Created by zpx on 2023/02/21.
//
#include <glog/logging.h>
#include <cstdint>
#include "ops/convolution_op.h"
#include "layer/layer_layer.h"
#include "layer/convolution_layer.h"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    ConvolutionLayer::ConvolutionLayer(const std::shared_ptr<RuntimeOperator> &op) : Layer("Convolution") {
        CHECK(op->op_type_ == OpType::kOperatorConvolution) << "Operator was a wrong type: " << int(op->op_type_);
        ConvolutionOperator *convolutionOperator = dynamic_cast<ConvolutionOperator *>(op.get());
        CHECK(convolutionOperator != nullptr) << "Convolution operator is empty";
        this->op_ = std::make_unique<ConvolutionOperator>(*convolutionOperator);
    }
/*
 * TODO: 重构卷积操作
 */
    void ConvolutionLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorConvolution);
        CHECK(!inputs.empty()) << "Input is empty";
        CHECK(inputs.size() == outputs.size());
        const auto &weight = this->op_->getWeight();
        CHECK(!weight.empty());
        std::vector<std::shared_ptr<ftensor >> bias_;
        if (this->op_->isUseBias()) {
            const auto &bias = this->op_->getBias();
            CHECK(!bias.empty());
            bias_ = bias;
        }
        const uint32_t stride_h = this->op_->getStride()[0];
        const uint32_t stride_w = this->op_->getStride()[1];
        CHECK(stride_h > 0 && stride_w > 0);
        const uint32_t padding_h = this->op_->getPadding()[0];
        const uint32_t padding_w = this->op_->getPadding()[1];
        const uint32_t groups = this->op_->getGroups();
        const uint32_t batch = inputs.size();
        CHECK(batch > 0);
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
//#pragma omp parallel for num_threads(batch)
        for (uint32_t i = 0; i < batch; i++) {
            const std::shared_ptr<ftensor> &input = inputs.at(i)->clone();
            CHECK(input != nullptr && !input->empty()) << "The input feature map of convolution layer is empty";
            std::shared_ptr<ftensor> input_;
            if (padding_h > 0 || padding_w > 0) {
                input_ = input;
                input_->padding({padding_h, padding_h, padding_w, padding_w}, 0);           // 四周填充全零
            } else {
                input_ = input;
            }
            const uint32_t input_w = input_->cols();
            const uint32_t input_h = input_->rows();
            const uint32_t input_c = input_->channels();
            const uint32_t kernel_count = weight.size();
            CHECK(kernel_count > 0) << "kernel count must greater than zero";
            uint32_t kernel_h = weight.at(0)->rows();
            uint32_t kernel_w = weight.at(0)->cols();
            CHECK(kernel_h > 0 && kernel_w) << "The size of kernel size is less than zero";
            if (groups != 1) {
                CHECK(kernel_count % groups == 0);
                CHECK(input_c % groups == 0);
            }
            uint32_t output_h = uint32_t(floor((input_h - kernel_h) / stride_h + 1));
            uint32_t output_w = uint32_t(floor((input_w - kernel_w) / stride_w + 1));
            for (uint32_t k = 0; k < kernel_count; ++k) {
                const std::shared_ptr<ftensor> &kernel = weight.at(k);
                CHECK(kernel->rows() == kernel_h);
                CHECK(kernel->cols() == kernel_w);
                CHECK(kernel->channels() == input_c / groups);
            }
            uint32_t row_len = kernel_w * kernel_h;
            uint32_t col_len = output_h * output_w;
            if (col_len == 0)col_len = 1;
            uint32_t input_c_group = input_c / groups;
            uint32_t kernel_count_group = kernel_count / groups;
            for (uint32_t g = 0; g < groups; ++g) {
                std::vector<arma::fmat> kernel_matrix_arr(kernel_count_group);
                arma::fmat kernel_matrix_c(1, row_len * input_c_group);
                for (uint32_t k = 0; k < kernel_count_group; ++k) {
                    const std::shared_ptr<ftensor> &kernel = weight.at(k + g * kernel_count_group);
                    for (uint32_t ic = 0; ic < input_c_group; ++ic) {
                        memcpy(kernel_matrix_c.memptr() + row_len * ic,
                               kernel->at(ic).memptr(), row_len * sizeof(float));
                    }
//#ifdef DEBUG
//                    LOG(INFO) << "kernel展开后: " << "\n" << kernel_matrix_c;
//#endif
                    kernel_matrix_arr.at(k) = kernel_matrix_c;
                }
                arma::fmat input_matrix(input_c_group * row_len, col_len);
                for (uint32_t ic = 0; ic < input_c_group; ++ic) {
                    const arma::fmat &input_channel = input_->at(ic + g * input_c_group);
                    int current_col = 0;
                    for (uint32_t w = 0; w < input_w - kernel_w + 1; w += stride_w) {
                        for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
                            float *input_matrix_c_ptr = input_matrix.colptr(current_col) + ic * row_len;
                            current_col++;
                            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                                const float *region_ptr = input_channel.colptr(w + kw) + r;
                                memcpy(input_matrix_c_ptr, region_ptr, kernel_h * sizeof(float));
                                input_matrix_c_ptr += kernel_h;
                            }
                        }
                    }
                }
//#ifdef DEBUG
//                LOG(INFO) << "input展开后: " << "\n" << input_matrix;
//#endif
                std::shared_ptr<ftensor> output_tensor = outputs.at(i);
                if (output_tensor == nullptr || outputs.empty()) {
                    output_tensor = std::make_shared<ftensor>(kernel_count, output_h, output_w);
                    outputs.at(i) = output_tensor;
                }
                CHECK(output_tensor->rows() == output_h && output_tensor->cols() == output_w &&
                      output_tensor->channels() == kernel_count) << "The output size of convolution is error";
                std::vector<arma::fmat> outputs_matrix(kernel_count_group);
                for (uint32_t k = 0; k < kernel_count_group; k++) {
                    const arma::fmat &output = kernel_matrix_arr.at(k) * input_matrix;
                    outputs_matrix.at(k) = output;
                }
                bool use_bias = this->op_->isUseBias();
                for (uint32_t k = 0; k < kernel_count_group; ++k) {
                    std::shared_ptr<ftensor> bias;
                    if (!bias_.empty() && use_bias)bias = bias_.at(k);
                    arma::fmat output = outputs_matrix.at(k);
                    CHECK(output.size() == output_h * output_w);
                    output.reshape(output_h, output_w);
                    if (bias != nullptr) {
                        float bias_value = bias->index(0);
                        output += bias_value;
                    }
                    output_tensor->at(k + g * kernel_count_group) = std::move(output);
                }
            }
        }
    }

    std::shared_ptr<Layer> ConvolutionLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op) {
        std::shared_ptr<Layer> convolutionLayer = std::make_shared<ConvolutionLayer>(op);
        return convolutionLayer;
    }

    void ConvolutionLayer::Forwards() {
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

    LayerRegistererWrapper convolutionRegisterWrapper(OpType::kOperatorConvolution, ConvolutionLayer::CreateInstance);
}