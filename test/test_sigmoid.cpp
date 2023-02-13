//
// Created by zpx on 2023/01/01.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/sigmoid_op.h"
#include "layer/sigmoid_layer.h"
#include "factory/layer_factory.hpp"
TEST(test_layer, forward_sigmoid1) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是3
    std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    inputs.push_back(input);
    SigmoidLayer layer(sigmoid_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 1/(1+ std::exp(1.f)));
        ASSERT_EQ(outputs.at(i)->index(1), 1/(1+ std::exp(2.f)));
        ASSERT_EQ(outputs.at(i)->index(2), 1/(1+ std::exp(-3.f)));
    }
}

TEST(test_layer, forward_sigmoid2) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();
    std::shared_ptr<Layer> sigmoid_layer = LayerRegisterer::CreateLayer(sigmoid_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    sigmoid_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 1/(1+ std::exp(1.f)));
        ASSERT_EQ(outputs.at(i)->index(1), 1/(1+ std::exp(2.f)));
        ASSERT_EQ(outputs.at(i)->index(2), 1/(1+ std::exp(-3.f)));
    }
}