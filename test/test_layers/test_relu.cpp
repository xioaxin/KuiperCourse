//
// Created by zpx on 2022/12/27.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.h"
#include "layer/relu_layer.h"

TEST(test_layer, forward_relu1) {
    using namespace kuiper_infer;
    float thresh = 0.f;
    std::shared_ptr<RuntimeOperator> relu_op = std::make_shared<ReluOperator>(thresh);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    ReluLayer layer(relu_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 3.f);
    }
}

TEST(test_layer, forward_relu2) {
    using namespace kuiper_infer;
    float thresh = 0.f;
    std::shared_ptr<RuntimeOperator> relu_op = std::make_shared<ReluOperator>(thresh);
    std::shared_ptr<Layer> relu_layer = LayerRegisterer::CreateLayer(relu_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    relu_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 3.f);
    }
}