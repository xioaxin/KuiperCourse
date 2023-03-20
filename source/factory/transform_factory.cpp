//
// Created by zpx on 2023/03/18.
//
#include "factory/transform_factory.h"
#include <glog/logging.h>

namespace kuiper_infer {
    void TransformFactory::forward(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs) {
        CHECK(!this->transforms_.empty()) << "The transform list is empty";
        CHECK(!inputs.empty()) << "The input data is empty";
        uint32_t batch_size = inputs.size();
        CHECK(inputs.size() == outputs.size()) << "The size of input data and output data is not equal";
        for (int i = 0; i < batch_size; ++i) {
            sftensor data = inputs.at(i);
            uint32_t channels = data->channels();
            uint32_t rows = data->rows();
            uint32_t cols = data->cols();
            for (auto &transform: transforms_) {
                data = transform->forward(data);
            }
            CHECK(data->channels() == channels) << "The input and output channel are not equal";
            CHECK(data->rows() == rows) << "The input and output row are not equal";
            CHECK(data->cols() == cols) << "The input and output cols are not equal";
            outputs[i] = data;
        }
    }
}