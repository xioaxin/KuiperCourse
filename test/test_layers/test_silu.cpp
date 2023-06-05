//
// Created by zpx on 2023/02/28.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/silu_op.h"
#include "layer/silu_layer.h"

TEST(test_layer, forward_silu1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> silu_op = std::make_shared<SiluOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是3
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    SiluLayer layer(silu_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), -1 / (1 + std::exp(1.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), -2 / (1 + std::exp(2.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 3 / (1 + std::exp(-3.0f)));
    }
}

TEST(test_layer, forward_silu2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> silu_op = std::make_shared<SiluOperator>();
    std::shared_ptr<Layer> sigmoid_layer = LayerRegisterer::CreateLayer(silu_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; i++) {
        inputs[i] = input;
    }
    sigmoid_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), -1 / (1 + std::exp(1.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), -2 / (1 + std::exp(2.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 3 / (1 + std::exp(-3.0f)));
    }
}

TEST(test_layer, forward_silu_cuda1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> silu_op = std::make_shared<SiluOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是3
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    SiluLayer layer(silu_op);
    layer.ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), -1 / (1 + std::exp(1.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), -2 / (1 + std::exp(2.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 3 / (1 + std::exp(-3.0f)));
    }
}

TEST(test_layer, forward_silu_cuda2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> silu_op = std::make_shared<SiluOperator>();
    std::shared_ptr<Layer> sigmoid_layer = LayerRegisterer::CreateLayer(silu_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; i++) {
        inputs[i] = input;
    }
    sigmoid_layer->ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), -1 / (1 + std::exp(1.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), -2 / (1 + std::exp(2.0f)));
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 3 / (1 + std::exp(-3.0f)));
    }
}