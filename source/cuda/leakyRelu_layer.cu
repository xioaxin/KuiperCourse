//
// Created by zpx on 2023/06/03.
//

#include <glog/logging.h>
#include "ops/leakyRelu_op.h"
#include "layer/leakyRelu_layer.h"
#include "helper/cuda_helper.cu"

namespace kuiper_infer {
    __global__ void leakyReluLayer_kernel(float *input, float *output, int N, float thresh, float alpha) {
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        if (index < N)output[index] = MAX_CUDA(input[index], thresh) + (MIN_CUDA(input[index], thresh)) * alpha;
    }

    void leakyReluLayer_gpu_kernel(const std::shared_ptr<Tensor<float>> &input, std::shared_ptr<Tensor<float>> &output,
                                   float thresh, float alpha) {
        float *dev_in, *dev_out;
        CUDA_CHECK(cudaMalloc(&dev_in, sizeof(float) * input->size()));
        CUDA_CHECK(cudaMalloc(&dev_out, sizeof(float) * input->size()));
        CUDA_CHECK(cudaMemcpy(dev_in, (float *) input->data().mem, sizeof(float) * input->size(), cudaMemcpyKind::cudaMemcpyHostToDevice));
        uint32_t bs = 256 < input->size() ? 256 : input->size();
        uint32_t ceil = (input->size() + bs - 1) / bs;
        leakyReluLayer_kernel<<<ceil, bs>>>(dev_in, dev_out, input->size(), thresh, alpha);
        CUDA_CHECK(cudaMemcpy(output->data().memptr(), dev_out, sizeof(float) * output->size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dev_in));
        CUDA_CHECK(cudaFree(dev_out));
    }

    void LeakyReluLayer::ForwardsCuda(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        CHECK(this->op_ != nullptr);
        CHECK(this->op_->op_type_ == OpType::kOperatorLeakyRelu);
        const uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The input size not equal with output size";
        for (int i = 0; i < batch_size; i++) {
            CHECK(!inputs.at(i)->empty());
            outputs.at(i) = std::make_shared<ftensor>(inputs.at(i)->shapes());
            leakyReluLayer_gpu_kernel(inputs.at(i), outputs.at(i), op_->get_thresh(), op_->getAlpha());
        }
    }
}