//
// Created by zpx on 2023/02/22.
//
//
// Created by fss on 23-2-2.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/ops.h"
#include "layer/convolution_layer.h"

// 单卷积单通道
TEST(test_layer, conv1) {
    using namespace kuiper_infer;
    LOG(INFO) << "My convolution test!";
    ConvolutionOperator *conv_op = new ConvolutionOperator(false, 1, 1, 1, 0, 0);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 3; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 3, 3);
    weight1->fill(values);
#ifdef DEBUG
    LOG(INFO) << "weight:";
    weight1->show();
#endif
    // 设置权重
    std::vector<sftensor> weights;
    weights.push_back(weight1);
    conv_op->setWeights(weights);
    std::shared_ptr<Operator> op = std::shared_ptr<ConvolutionOperator>(conv_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4;"
                            "5,6,7,8;"
                            "7,8,9,10;"
                            "11,12,13,14";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 4, 4);
    input->at(0) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    ConvolutionLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);
    layer.Forwards(inputs, outputs);
    LOG(INFO) << "result: ";
    ASSERT_EQ(outputs[0]->at(0, 0, 0), 114);
    ASSERT_EQ(outputs[0]->at(0, 0, 1), 132);
    ASSERT_EQ(outputs[0]->at(0, 1, 0), 174);
    ASSERT_EQ(outputs[0]->at(0, 1, 1), 192);
}

// 多卷积多通道
TEST(test_layer, conv2) {
    using namespace kuiper_infer;
    LOG(INFO) << "My convolution test!";
    ConvolutionOperator *conv_op = new ConvolutionOperator(false, 1, 1, 1, 0, 0);
    // 单个卷积核的情况
    std::vector<float> values;
    arma::fmat weight_data = "1 ,1, 1 ;"
                             "2 ,2, 2;"
                             "3 ,3, 3;";
    // 初始化三个卷积核
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 3);
    weight1->at(0) = weight_data;
    weight1->at(1) = weight_data;
    weight1->at(2) = weight_data;
    std::shared_ptr<ftensor> weight2 = weight1->clone();
    std::shared_ptr<ftensor> weight3 = weight1->clone();
#ifdef DEBUG
    LOG(INFO) << "weight:";
    weight1->show();
#endif
    // 设置权重
    std::vector<sftensor> weights;
    weights.push_back(weight1);
    weights.push_back(weight2);
    weights.push_back(weight3);
    conv_op->setWeights(weights);
    std::shared_ptr<Operator> op = std::shared_ptr<ConvolutionOperator>(conv_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4;"
                            "5,6,7,8;"
                            "7,8,9,10;"
                            "11,12,13,14";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;
#ifdef DEBUG
    LOG(INFO) << "input:";
    input->show();
#endif
    // 权重数据和输入数据准备完毕
    for (int i = 0; i < MAX_TEST_ITERATION; ++i) {
        inputs.push_back(input);
    }
    ConvolutionLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(MAX_TEST_ITERATION);
    layer.Forwards(inputs, outputs);
    LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i) {
#ifdef DEBUG
        outputs[i]->show();
#endif
        ASSERT_EQ(outputs[i]->at(0, 0, 0), 342);
        ASSERT_EQ(outputs[i]->at(0, 0, 1), 396);
        ASSERT_EQ(outputs[i]->at(0, 1, 0), 522);
        ASSERT_EQ(outputs[i]->at(0, 1, 1), 576);
    }
}