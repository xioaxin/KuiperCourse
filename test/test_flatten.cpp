//
// Created by zpx on 2023/02/26.
//
#include "layer/flatten_layer.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/flatten_op.h"

TEST(test_layer, flatten1) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> flatten_op = std::make_shared<FlattenOperator>(0, 1);
    std::vector<sftensor> inputs;
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 3, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;
    inputs.push_back(input);
    inputs.push_back(input);
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    FlattenLayer layer(flatten_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 3);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs[i]->shapes(), std::vector<uint32_t>({1, 9, 3}));
        ASSERT_EQ(outputs[i]->index(2), 7);
        ASSERT_EQ(outputs[i]->index(6), 1);
        ASSERT_EQ(outputs[i]->index(9), 2);
    }
}

TEST(test_layer, flatten2) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> flatten_op = std::make_shared<FlattenOperator>(0, 2);
    std::vector<sftensor> inputs;
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 3, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;
    inputs.push_back(input);
    inputs.push_back(input);
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    FlattenLayer layer(flatten_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 3);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs[i]->shapes(), std::vector<uint32_t>({1, 27, 1}));
        ASSERT_EQ(outputs[i]->index(2), 3);
        ASSERT_EQ(outputs[i]->index(6), 7);
        ASSERT_EQ(outputs[i]->index(9), 1);
    }
}