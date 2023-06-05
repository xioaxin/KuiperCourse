//
// Created by zpx on 2023/02/26.
//
#include "layer/hardSigmoid_layer.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(test_layer, forward_hardSigmoid1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardSigmoid_op = std::make_shared<HardSigmoidOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    HardSigmoidLayer layer(hardSigmoid_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
//        outputs.at(i)->show();
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 1.f);
    }
}

TEST(test_layer, forward_hardSigmoid2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardSigmoid_op = std::make_shared<HardSigmoidOperator>();
    std::shared_ptr<Layer> hardSigmoid_layer = LayerRegisterer::CreateLayer(hardSigmoid_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    hardSigmoid_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
//        outputs.at(i)->show();
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 1.f);
    }
}

TEST(test_layer, forward_hardSigmoid_cuda1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardSigmoid_op = std::make_shared<HardSigmoidOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    HardSigmoidLayer layer(hardSigmoid_op);
    layer.ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 1.f);
    }
}

TEST(test_layer, forward_hardSigmoid_cuda2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardSigmoid_op = std::make_shared<HardSigmoidOperator>();
    std::shared_ptr<Layer> hardSigmoid_layer = LayerRegisterer::CreateLayer(hardSigmoid_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 3.f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    hardSigmoid_layer->ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 1.f);
    }
}