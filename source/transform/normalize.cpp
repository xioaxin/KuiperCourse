//
// Created by zpx on 2023/03/18.
//
#include "transform/normalize.h"
#include <glog/logging.h>

namespace kuiper_infer {
    Normalize::Normalize(const std::vector<float> &mean, const std::vector<float> &std, const float max_pixel_value)
            : TransformBase("Normalize"), mean_(mean), std_(std), max_pixel_value_(255.0) {};

    sftensor Normalize::forward(kuiper_infer::sftensor &inputs) {
        CHECK(!inputs->empty()) << "The input of normalize transform is emtpy";
        int channels = inputs->channels();
        CHECK(std_.size() == mean_.size()) << "The normalize std and mean size are not equal";
        CHECK_EQ(channels, mean_.size()) << "The input channels and mean size are not equal";
        sftensor data = inputs->clone();
        sftensor mean = std::make_shared<ftensor>(data->shapes());
        sftensor std = std::make_shared<ftensor>(data->shapes());
        for (int i = 0; i < channels; ++i) {
            mean->at(i).fill(max_pixel_value_ * mean_[i]);
            std->at(i).fill(max_pixel_value_ * std_[i]);
        }
        data = Tensor<float>::elementDiv(std, Tensor<float>::elementSub(mean, data));
        return data;
    }
}