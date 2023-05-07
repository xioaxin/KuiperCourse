//
// Created by zpx on 2023/02/26.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/runtime_op.h"
#include "ops/sigmoid_op.h"
#include "layer/cat_layer.h"
//单个batch
TEST(test_layer, forward_cat1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> cat_op = std::make_shared<CatOperator>(1);
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
    inputs.push_back(input);
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    outputs.push_back(std::make_shared<ftensor>(15, 3, 3));
    CatLayer layer(cat_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[i]->at(0, 0, 0), 1);
        ASSERT_EQ(outputs[i]->at(1, 0, 2), 3);
        ASSERT_EQ(outputs[i]->at(2, 2, 2), 9);
    }
}
// 多个batch
TEST(test_layer, forward_cat2) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> cat_op = std::make_shared<CatOperator>(1);
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
    inputs.push_back(input);
    std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
    outputs.push_back(std::make_shared<ftensor>(12, 3, 3));
    CatLayer layer(cat_op);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    outputs[0]->show();
#endif
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[0]->at(i, 0, 0), 1);
        ASSERT_EQ(outputs[0]->at(i, 0, 2), 3);
        ASSERT_EQ(outputs[0]->at(i, 2, 2), 9);
    }
}