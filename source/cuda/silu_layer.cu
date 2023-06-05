//
// Created by zpx on 2023/06/03.
//

#include <glog/logging.h>
#include "ops/silu_op.h"
#include "layer/silu_layer.h"
#include "helper/cuda_helper.cu"

namespace kuiper_infer {
    __global__ void siluLayer_kernel(float *input, float *output, int N) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < N)output[index] = input[index] / (1.0 + exp(-input[index]));
    }

    void siluLayer_gpu_kernel(const std::shared_ptr<Tensor<float>> &input, std::shared_ptr<Tensor<float>> &output) {
        float *dev_in, *dev_out;
        CUDA_CHECK(cudaMalloc(&dev_in, sizeof(float) * input->size()));
        CUDA_CHECK(cudaMalloc(&dev_out, sizeof(float) * input->size()));
        CUDA_CHECK(cudaMemcpy(dev_in, input->data().memptr(), sizeof(float) * input->size(), cudaMemcpyKind::cudaMemcpyHostToDevice));
        uint32_t bs = 256 < input->size() ? 256 : input->size();
        uint32_t ceil = (input->size() + bs - 1) / bs;
        siluLayer_kernel<<<ceil, bs>>>(dev_in, dev_out, input->size());
        CUDA_CHECK(cudaMemcpy(output->data().memptr(), dev_out, sizeof(float) * output->size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dev_in));
        CUDA_CHECK(cudaFree(dev_out));
    }

    void SiluLayer::ForwardsCuda(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorSilu);
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            outputs.at(i) = std::make_shared<ftensor>(inputs.at(i)->shapes());
            siluLayer_gpu_kernel(inputs.at(i), outputs.at(i));
        }
    }
}