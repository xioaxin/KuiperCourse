//
// Created by zpx on 2023/02/22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/ops.h"
#include "layer/batchNorm_layer.h"
// 单通道正则化层计算
TEST(test_layer, batchNorm1) {
    using namespace kuiper_infer;
    BatchNormOperator *batch_norm_operator = new BatchNormOperator(0);
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
    std::shared_ptr<Operator> op = std::shared_ptr<BatchNormOperator>(batch_norm_operator);
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
    std::vector<sftensor> outputs;
    outputs.push_back(output);
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
    BatchNormOperator *batch_norm_operator = new BatchNormOperator(0);
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
    std::shared_ptr<Operator> op = std::shared_ptr<BatchNormOperator>(batch_norm_operator);
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
    inputs.push_back(input);
    BatchNormLayer layer(op);
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 3, 3);
    std::vector<sftensor> outputs;
    outputs.push_back(output);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
    outputs[0]->show();
#endif
    ASSERT_EQ(outputs[0]->at(0, 0, 0), 1);
    ASSERT_EQ(outputs[0]->at(0, 0, 1), 2);
    ASSERT_EQ(outputs[0]->at(1, 1, 0), 5);
    ASSERT_EQ(outputs[0]->at(1, 1, 1), 6);
    ASSERT_EQ(outputs[0]->at(2, 2, 1), 8);
    ASSERT_EQ(outputs[0]->at(2, 2, 2), 9);
}
// 多卷积多通道
//TEST(test_layer, conv2) {
//    using namespace kuiper_infer;
//    LOG(INFO) << "My convolution test!";
//    ConvolutionOperator *conv_op = new ConvolutionOperator(false, 1, 1, 1, 0, 0);
//    // 单个卷积核的情况
//    std::vector<float> values;
//    arma::fmat weight_data = "1 ,1, 1 ;"
//                             "2 ,2, 2;"
//                             "3 ,3, 3;";
//    // 初始化三个卷积核
//    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 3);
//    weight1->at(0) = weight_data;
//    weight1->at(1) = weight_data;
//    weight1->at(2) = weight_data;
//    std::shared_ptr<ftensor> weight2 = weight1->clone();
//    std::shared_ptr<ftensor> weight3 = weight1->clone();
//    LOG(INFO) << "weight:";
//    weight1->show();
//    // 设置权重
//    std::vector<sftensor> weights;
//    weights.push_back(weight1);
//    weights.push_back(weight2);
//    weights.push_back(weight3);
//    conv_op->setWeights(weights);
//    std::shared_ptr<Operator> op = std::shared_ptr<ConvolutionOperator>(conv_op);
//    std::vector<std::shared_ptr<ftensor >> inputs;
//    arma::fmat input_data = "1,2,3,4;"
//                            "5,6,7,8;"
//                            "7,8,9,10;"
//                            "11,12,13,14";
//    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);
//    input->at(0) = input_data;
//    input->at(1) = input_data;
//    input->at(2) = input_data;
//    LOG(INFO) << "input:";
//    input->show();
//    // 权重数据和输入数据准备完毕
//    inputs.push_back(input);
//    ConvolutionLayer layer(op);
//    std::vector<std::shared_ptr<ftensor >> outputs(1);
//    layer.Forwards(inputs, outputs);
//    LOG(INFO) << "result: ";
//    for (int i = 0; i < outputs.size(); ++i) {
//        outputs.at(i)->show();
//    }
//}