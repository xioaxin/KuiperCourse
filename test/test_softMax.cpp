//
// Created by zpx on 2022/12/27.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/softMax_op.h"
#include "layer/softMax_layer.h"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_softMax1) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> softMax_op = std::make_shared<SoftMaxOperator>();
    arma::fmat input_data1 = "1,2,3,4;"
                             "5,6,7,8;"
                             "1,2,3,4;"
                             "5,6,7,8";
    arma::fmat input_data2 = "1,2,3,4;"
                             "7,2,3,4;"
                             "9,8,7,6;"
                             "5,4,3,2";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(2, 4, 4);
    input->at(0) = input_data1;
    input->at(1) = input_data2;
    std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
    inputs.push_back(input);
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(2, 4, 4);
    std::vector<sftensor> outputs;
    outputs.push_back(output);
    SoftMaxLayer layer(softMax_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    output = outputs.at(0);
#ifdef DEBUG
    output->show();
#endif
    ASSERT_EQ(output->at(0, 0, 0), 0.5f);
    ASSERT_EQ(output->at(1, 0, 0), 0.5f);
}
