//
// Created by zpx on 2023/02/24.
//
#include "runtime/runtime_ir.h"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_forward, forward1) {
    using namespace kuiper_infer;
    const std::string &param_path = "../tmp/resnet18_hub.pnnx.param";
    const std::string &weight_path = "../tmp/resnet18_hub.pnnx.bin";
    RuntimeGraph graph(param_path, weight_path);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &operators = graph.operators();
    LOG(INFO) << "operator size: " << operators.size();
    uint32_t batch_size = 2;
    std::vector<sftensor> inputs(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
        inputs.at(i) = std::make_shared<ftensor>(3, 224, 224);
        inputs.at(i)->fill(1.f);
    }
    const std::vector<sftensor> &outputs = graph.forward(inputs, true);
    ASSERT_EQ(outputs[0]->size(), 1000);
}