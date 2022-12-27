//
// Created by zpx on 2022/12/26.
//
#include "layer/layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
//    explicit Layer(const std::string &layer_name);
    Layer::Layer( const std::string &layer_name) : layer_name_(layer_name) {}

    void Layer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        LOG(FATAL) << "The layer " << this->layer_name_ << " not implement yet!";
    }
}