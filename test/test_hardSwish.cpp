//
// Created by zpx on 2023/02/26.
//
#include "layer/hardSwish_layer.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(test_layer, forward_hardSwish1) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> hardHardSwish_op = std::make_shared<HardSwishOperator>();
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = 9.f; //output对应的应该是0
    input->index(1) = -3.f; //output对应的应该是0
    input->index(2) = 6.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    inputs.push_back(input);
    HardSwishLayer layer(hardHardSwish_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs.at(i)->index(0), 9.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 6.f);
    }
}

TEST(test_layer, forward_hardSwish2) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> hardSwish_op = std::make_shared<HardSwishOperator>();
    std::shared_ptr<Layer> HardSwishLayer = LayerRegisterer::CreateLayer(hardSwish_op);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
    input->index(0) = 9.f; //output对应的应该是0
    input->index(1) = -3.f; //output对应的应该是0
    input->index(2) = 6.f; //output对应的应该是1
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    inputs.push_back(input);
    inputs.push_back(input);
    HardSwishLayer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 3);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs.at(i)->index(0), 9.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 6.f);
    }
}