//
// Created by zpx on 2022/12/26.
//
#include "layer/layer_layer.h"
#include <glog/logging.h>

namespace kuiper_infer {
    Layer::Layer(const std::string &layer_name) : layer_name_(layer_name) {}
}