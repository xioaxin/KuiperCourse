//
// Created by zpx on 2023/06/03.
//

#include <glog/logging.h>
#include "ops/adaptiveMaxPooling_op.h"
#include "layer/adaptiveMaxPooling_layer.h"
#include "helper/cuda_helper.cu"

namespace kuiper_infer {
    __global__ void adaptiveMaxPoolingLayer_kernel(float *input, float *output, int N) {
    }

    void adaptiveMaxPoolingLayer_gpu_kernel(const std::shared_ptr<Tensor<float>> &input, std::shared_ptr<Tensor<float>> &output) {
//        float *dev_in, *dev_out;
//        CUDA_CHECK(cudaMalloc(&dev_in, sizeof(float) * input->size()));
//        CUDA_CHECK(cudaMalloc(&dev_out, sizeof(float) * input->size()));
//        CUDA_CHECK(cudaMemcpy(dev_in, (float *) input->data().mem, sizeof(float) * input->size(), cudaMemcpyKind::cudaMemcpyHostToDevice));
//        uint32_t bs = 1024;
//        uint32_t ceil = (input->size() + bs - 1) / bs;
//        adaptiveAvgPoolingLayer_kernel<<<ceil, bs>>>(dev_in, dev_out, input->size());
//        CUDA_CHECK(
//                cudaMemcpy((float *) output->data().mem, dev_out, sizeof(float) * output->size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }

    void AdaptiveMaxPoolingLayer::ForwardsCuda(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
//        CHECK(this->op_ != nullptr);
//        CHECK(this->op_->op_type_ == OpType::kOperatorRelu);
//        const uint32_t batch_size = inputs.size();
//        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
//        for (int i = 0; i < batch_size; i++) {
//            CHECK(!inputs.at(i)->empty());
//            adaptiveMaxPoolingLayer_gpu_kernel(inputs.at(i), outputs.at(i));
//        }
    }
}