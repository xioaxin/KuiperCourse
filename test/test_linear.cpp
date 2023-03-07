//
// Created by zpx on 2023/02/28.
//
//
// Created by zpx on 2023/02/22.
//
//
// Created by fss on 23-2-2.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/ops.h"
#include "layer/linear_layer.h"

// 单通道线性层+偏置
TEST(test_layer, test_linear1_bias) {
    using namespace kuiper_infer;
    LinearOperator *linear_op = new LinearOperator(5, 6);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 5; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 5, 6);
    weight1->fill(values);
#ifdef DEBUG
    LOG(INFO) << "weight:";
    weight1->show();
#endif
    // 设置权重
    linear_op->setWeights(weight1);
    std::vector<float> value_bias;
    for (int i = 0; i < 3; ++i) {
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
    }
    std::shared_ptr<ftensor> bias = std::make_shared<ftensor>(1, 3, 6);
    bias->fill(value_bias);
#ifdef DEBUG
    LOG(INFO) << "bias:";
    bias->show();
#endif
    linear_op->setBias(bias);
    linear_op->setUseBias(true);
    std::shared_ptr<Operator> op = std::shared_ptr<LinearOperator>(linear_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4,5;"
                            "5,6,7,8,9;"
                            "7,8,9,10,11;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 3, 5);
    input->at(0) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    LinearLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
#endif
    ASSERT_EQ(outputs[0]->at(0, 0, 0), 56);
    ASSERT_EQ(outputs[0]->at(0, 1, 0), 116);
    ASSERT_EQ(outputs[0]->at(0, 2, 0), 146);
}



// 单通道线性层
TEST(test_layer, test_linear1) {
    using namespace kuiper_infer;
    LinearOperator *linear_op = new LinearOperator(5, 6);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 5; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 5, 6);
    weight1->fill(values);
#ifdef DEBUG
    LOG(INFO) << "weight:";
    weight1->show();
#endif
    // 设置权重
    linear_op->setWeights(weight1);
    std::shared_ptr<Operator> op = std::shared_ptr<LinearOperator>(linear_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4,5;"
                            "5,6,7,8,9;"
                            "7,8,9,10,11;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 3, 5);
    input->at(0) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    LinearLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
#endif
    ASSERT_EQ(outputs[0]->at(0, 0, 0), 55);
    ASSERT_EQ(outputs[0]->at(0, 1, 0), 115);
    ASSERT_EQ(outputs[0]->at(0, 2, 0), 145);
}



// 多batch线性层+偏置
TEST(test_layer, test_linear2_bias) {
    using namespace kuiper_infer;
    LinearOperator *linear_op = new LinearOperator(5, 6);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 5; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 5, 6);
    weight1->fill(values);
#ifdef DEBUG
    LOG(INFO) << "weight:";
    weight1->show();
#endif
    // 设置权重
    linear_op->setWeights(weight1);
    std::vector<float> value_bias;
    for (int i = 0; i < 3; ++i) {
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
        value_bias.push_back(float(1));
    }
    std::shared_ptr<ftensor> bias = std::make_shared<ftensor>(1, 3, 6);
    bias->fill(value_bias);
#ifdef DEBUG
    LOG(INFO) << "bias:";
    bias->show();
#endif
    linear_op->setBias(bias);
    linear_op->setUseBias(true);
    std::shared_ptr<Operator> op = std::shared_ptr<LinearOperator>(linear_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4,5;"
                            "5,6,7,8,9;"
                            "7,8,9,10,11;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 3, 5);
    input->at(0) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    for (int i = 0; i < 10; ++i) {
        inputs.push_back(input);
    }
    linear_op->setUseBias(true);
    LinearLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
#endif
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[i]->at(0, 0, 0), 56);
        ASSERT_EQ(outputs[i]->at(0, 1, 0), 116);
        ASSERT_EQ(outputs[i]->at(0, 2, 0), 146);
    }
}

// 多batch线性层
TEST(test_layer, test_linear3) {
    using namespace kuiper_infer;
    LinearOperator *linear_op = new LinearOperator(5, 6);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 5; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 5, 6);
    weight1->fill(values);
#ifdef DEBUG
    LOG(INFO) << "weight:";
    weight1->show();
#endif
    // 设置权重
    linear_op->setWeights(weight1);
    std::shared_ptr<Operator> op = std::shared_ptr<LinearOperator>(linear_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4,5;"
                            "5,6,7,8,9;"
                            "7,8,9,10,11;";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 3, 5);
    input->at(0) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    inputs.push_back(input);
    inputs.push_back(input);
    LinearLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);
    layer.Forwards(inputs, outputs);
#ifdef DEBUG
    LOG(INFO) << "result: ";
#endif
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs[i]->at(0, 0, 0), 55);
        ASSERT_EQ(outputs[i]->at(0, 1, 0), 115);
        ASSERT_EQ(outputs[i]->at(0, 2, 0), 145);
    }
}