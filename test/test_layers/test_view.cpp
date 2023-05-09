//
// Created by zpx on 2023/05/07.
//
#include "layer/view_layer.h"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_layer, view1) {
    using namespace kuiper_infer;
    std::vector<int> shape{2, -1};
    std::shared_ptr<RuntimeOperator> view_op = std::make_shared<ViewOperator>(shape);
    std::vector<sftensor> inputs(1);
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(2, 3, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    inputs[0] = input;
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    ViewLayer layer(view_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs[i]->shapes(), std::vector<uint32_t>({1, 2, 9}));
        ASSERT_EQ(outputs[i]->at(0, 1, 1), 2);
        ASSERT_EQ(outputs[i]->at(0, 0, 3), 5);
        ASSERT_EQ(outputs[i]->at(0, 1, 6), 7);
    }
}

TEST(test_layer, view2) {
    using namespace kuiper_infer;
    std::vector<int> shape{-1, 2};
    std::shared_ptr<RuntimeOperator> flatten_op = std::make_shared<ViewOperator>(shape);
    std::vector<sftensor> inputs(MAX_TEST_ITERATION);
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(2, 3, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs[i] = input;
    }
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION); //放结果
    ViewLayer layer(flatten_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs[i]->shapes(), std::vector<uint32_t>({1, 9, 2}));
        ASSERT_EQ(outputs[i]->at(0, 1, 1), 5);
        ASSERT_EQ(outputs[i]->at(0, 0, 1), 2);
        ASSERT_EQ(outputs[i]->at(0, 8, 1), 9);
    }
}

TEST(test_layer, view3) {
    using namespace kuiper_infer;
    std::vector<int> shape{1, 9, 3};
    std::shared_ptr<RuntimeOperator> view_op = std::make_shared<ViewOperator>(shape);
    std::vector<sftensor> inputs(1);
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 3, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;
    inputs[0] = input;
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
    ViewLayer layer(view_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs[i]->shapes(), std::vector<uint32_t>({1, 9, 3}));
        ASSERT_EQ(outputs[i]->at(0, 1, 1), 6);
        ASSERT_EQ(outputs[i]->at(0, 5, 2), 9);
        ASSERT_EQ(outputs[i]->at(0, 6, 1), 2);
    }
}