//
// Created by zpx on 2022/12/27.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu6_op.h"
#include "layer/relu6_layer.h"

TEST(test_layer, forward_relu6_1) {
    using namespace kuiper_infer;
    float thresh = 1.f;
    std::shared_ptr<RuntimeOperator> relu6_op = std::make_shared<Relu6Operator>(thresh);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是3
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    Relu6Layer layer(relu6_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), std::min(std::max(-1.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(1), std::min(std::max(-2.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(2), std::min(std::max(3.f, 0.f), thresh));
    }
}

TEST(test_layer, forward_relu6_2) {
    using namespace kuiper_infer;
    float thresh = 0.f;
    std::shared_ptr<RuntimeOperator> relu6_op = std::make_shared<Relu6Operator>(thresh);
    std::shared_ptr<Layer> relu6_layer = LayerRegisterer::CreateLayer(relu6_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    relu6_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), std::min(std::max(-1.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(1), std::min(std::max(-2.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(2), std::min(std::max(3.f, 0.f), thresh));
    }
}

TEST(test_layer, forward_relu6_cuda_1) {
    using namespace kuiper_infer;
    float thresh = 1.f;
    std::shared_ptr<RuntimeOperator> relu6_op = std::make_shared<Relu6Operator>(thresh);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是3
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    Relu6Layer layer(relu6_op);
    layer.ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), std::min(std::max(-1.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(1), std::min(std::max(-2.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(2), std::min(std::max(3.f, 0.f), thresh));
    }
}

TEST(test_layer, forward_relu6_cuda_2) {
    using namespace kuiper_infer;
    float thresh = 0.f;
    std::shared_ptr<RuntimeOperator> relu6_op = std::make_shared<Relu6Operator>(thresh);
    std::shared_ptr<Layer> relu6_layer = LayerRegisterer::CreateLayer(relu6_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    relu6_layer->ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), std::min(std::max(-1.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(1), std::min(std::max(-2.f, 0.f), thresh));
        ASSERT_EQ(outputs.at(i)->index(2), std::min(std::max(3.f, 0.f), thresh));
    }
}