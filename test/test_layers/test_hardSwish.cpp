//
// Created by zpx on 2023/02/26.
//
#include "layer/hardSwish_layer.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(test_layer, forward_hardSwish1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardHardSwish_op = std::make_shared<HardSwishOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = 9.f;  // output对应的应该是0
    input->index(1) = -3.f; // output对应的应该是0
    input->index(2) = 6.f;  // output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1);   // 作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);  // 放结果
    inputs[0] = input;
    HardSwishLayer layer(hardHardSwish_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), 9.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 6.f);
    }
}

TEST(test_layer, forward_hardSwish2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardSwish_op = std::make_shared<HardSwishOperator>();
    std::shared_ptr<Layer> HardSwishLayer = LayerRegisterer::CreateLayer(hardSwish_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = 9.f; //output对应的应该是0
    input->index(1) = -3.f; //output对应的应该是0
    input->index(2) = 6.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    HardSwishLayer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), 9.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 6.f);
    }
}

TEST(test_layer, forward_hardSwish_cuda1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardHardSwish_op = std::make_shared<HardSwishOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = 9.f; //output对应的应该是0
    input->index(1) = -3.f; //output对应的应该是0
    input->index(2) = 6.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1); //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    inputs[0] = input;
    HardSwishLayer layer(hardHardSwish_op);
    layer.ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), 9.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 6.f);
    }
}

TEST(test_layer, forward_hardSwish_cuda2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> hardSwish_op = std::make_shared<HardSwishOperator>();
    std::shared_ptr<Layer> HardSwishLayer = LayerRegisterer::CreateLayer(hardSwish_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = 9.f; //output对应的应该是0
    input->index(1) = -3.f; //output对应的应该是0
    input->index(2) = 6.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    HardSwishLayer->ForwardsCuda(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_FLOAT_EQ(outputs.at(i)->index(0), 9.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_FLOAT_EQ(outputs.at(i)->index(2), 6.f);
    }
}