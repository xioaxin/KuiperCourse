//
// Created by zpx on 2023/02/22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/runtime_op.h"
#include "layer/batchNorm_layer.h"
// 单通道正则化层计算
TEST(test_layer, batchNorm1) {
    using namespace kuiper_infer;
    std::shared_ptr<BatchNormOperator> batch_norm_operator = std::make_shared<BatchNormOperator>(0);
    // 设置均值
    std::shared_ptr<ftensor> mean_value_weight = std::make_shared<ftensor>(1, 1, 1);
    mean_value_weight->fill(0);
#ifdef DEBUG
    LOG(INFO) << "mean:";
    mean_value_weight->show();
#endif
    batch_norm_operator->setMeanValue(mean_value_weight);
    // 设置方差
    std::shared_ptr<ftensor> var_value_weight = std::make_shared<ftensor>(1, 1, 1);
    var_value_weight->fill(1);
#ifdef DEBUG
    LOG(INFO) << "var:";
    var_value_weight->show();
#endif
    batch_norm_operator->setVarValue(var_value_weight);
    // 设置权重
    std::vector<float> affine_alpha_value = {1};
    batch_norm_operator->setAffineAlpha(affine_alpha_value);
    // 设置偏置项
    std::vector<float> affine_beta_value = {0};
    batch_norm_operator->setAffineBata(affine_beta_value);
    std::shared_ptr<RuntimeOperator> op = std::shared_ptr<BatchNormOperator>(batch_norm_operator);
    std::vector<sftensor> inputs;
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 3, 3);
    input->at(0) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    BatchNormLayer layer(op);
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(1, 3, 3);
    std::vector<sftensor> outputs(1);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
    outputs[0]->show();
#endif
    ASSERT_EQ(outputs[0]->at(0, 0, 0), 1);
    ASSERT_EQ(outputs[0]->at(0, 0, 1), 2);
    ASSERT_EQ(outputs[0]->at(0, 1, 0), 5);
    ASSERT_EQ(outputs[0]->at(0, 1, 1), 6);
    ASSERT_EQ(outputs[0]->at(0, 2, 1), 8);
    ASSERT_EQ(outputs[0]->at(0, 2, 2), 9);
}
// 多通道正则化层计算
TEST(test_layer, batchNorm2) {
    using namespace kuiper_infer;
    std::shared_ptr<BatchNormOperator> batch_norm_operator = std::make_shared<BatchNormOperator>(0);
    // 设置均值
    std::shared_ptr<ftensor> mean_value_weight = std::make_shared<ftensor>(3, 1, 1);
    mean_value_weight->fill(0);
#ifdef DEBUG
    LOG(INFO) << "mean:";
    mean_value_weight->show();
#endif
    batch_norm_operator->setMeanValue(mean_value_weight);
    // 设置方差
    std::shared_ptr<ftensor> var_value_weight = std::make_shared<ftensor>(3, 1, 1);
    var_value_weight->fill(1);
#ifdef DEBUG
    LOG(INFO) << "var:";
    var_value_weight->show();
#endif
    batch_norm_operator->setVarValue(var_value_weight);
    // 设置权重
    std::vector<float> affine_alpha_value = {1, 1, 1};
    batch_norm_operator->setAffineAlpha(affine_alpha_value);
    // 设置偏置项
    std::vector<float> affine_beta_value = {0, 0, 0};
    batch_norm_operator->setAffineBata(affine_beta_value);
    std::shared_ptr<RuntimeOperator> op = std::shared_ptr<BatchNormOperator>(batch_norm_operator);
    std::vector<sftensor> inputs;
    arma::fmat input_data = "1,2,3;"
                            "5,6,7;"
                            "7,8,9;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 3, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    BatchNormLayer layer(op);
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 3, 3);
    std::vector<sftensor> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs.push_back(input);
    }
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
    outputs[0]->show();
#endif
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[i]->at(0, 0, 0), 1);
        ASSERT_EQ(outputs[i]->at(0, 0, 1), 2);
        ASSERT_EQ(outputs[i]->at(1, 1, 0), 5);
        ASSERT_EQ(outputs[i]->at(1, 1, 1), 6);
        ASSERT_EQ(outputs[i]->at(2, 2, 1), 8);
        ASSERT_EQ(outputs[i]->at(2, 2, 2), 9);
    }
}