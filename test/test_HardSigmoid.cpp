//
// Created by zpx on 2023/02/26.
//
#include "layer/hardSigmoid_layer.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(test_layer, forward_hardSigmoid1) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> hardSigmoid_op = std::make_shared<HardSigmoidOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f; //output对应的应该是0
    input->index(1) = -2.f; //output对应的应该是0
    input->index(2) = 3.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    inputs.push_back(input);
    HardSigmoidLayer layer(hardSigmoid_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 1.f);
    }
}

TEST(test_layer, forward_hardSigmoid2) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> hardSigmoid_op = std::make_shared<HardSigmoidOperator>();
    std::shared_ptr<Layer> hardSigmoid_layer = LayerRegisterer::CreateLayer(hardSigmoid_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = -1.f;
    input->index(1) = -2.f;
    input->index(2) = 0.5f;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    hardSigmoid_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 0.75f);
    }
}