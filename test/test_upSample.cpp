//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/ops.h"
#include "ops/upSample_op.h"
#include "layer/layer.h"
#include "factory/layer_factory.hpp"

TEST(test_layer, upsample_Layer1) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> upSample_op = std::make_shared<UpSampleOperator>(2.0, 2.0, UpSampleMode::kModelNearest);
    std::shared_ptr<Layer> upSample_layer = LayerRegisterer::CreateLayer(upSample_op);
    CHECK(upSample_layer != nullptr);
    arma::fmat input_data = "1 2;"
                            "3 4;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    upSample_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
#ifdef DEBUG
    outputs[0]->show();
#endif
    const auto &output = outputs.at(0);
    ASSERT_EQ(output->rows(), 4);
    ASSERT_EQ(output->cols(), 4);
    ASSERT_EQ(output->at(0, 0, 0), 1);
    ASSERT_EQ(output->at(0, 0, 2), 2);
    ASSERT_EQ(output->at(0, 2, 1), 3);
    ASSERT_EQ(output->at(0, 2, 3), 4);
    ASSERT_EQ(output->at(0, 3, 1), 3);
}

TEST(test_layer, upsample_Layer2) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> upSample_op = std::make_shared<UpSampleOperator>(2.0, 2.0, UpSampleMode::kModelNearest);
    std::shared_ptr<Layer> upSample_layer = LayerRegisterer::CreateLayer(upSample_op);
    CHECK(upSample_layer != nullptr);
    arma::fmat input_data = "1,2,5;"
                            "3,4,6;"
                            "7,8,9";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    upSample_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
#ifdef DEBUG
    outputs[0]->show();
#endif
    const auto &output = outputs.at(0);
    ASSERT_EQ(output->rows(), 6);
    ASSERT_EQ(output->cols(), 6);
    ASSERT_EQ(output->at(0, 0, 0), 1);
    ASSERT_EQ(output->at(0, 0, 2), 2);
    ASSERT_EQ(output->at(0, 2, 1), 3);
    ASSERT_EQ(output->at(0, 2, 3), 4);
    ASSERT_EQ(output->at(0, 3, 1), 3);
    ASSERT_EQ(output->at(0, 4, 1), 7);
}

TEST(test_layer, upsample_Layer3) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> upSample_op = std::make_shared<UpSampleOperator>(2.0, 2.0, UpSampleMode::kModelNearest);
    std::shared_ptr<Layer> upSample_layer = LayerRegisterer::CreateLayer(upSample_op);
    CHECK(upSample_layer != nullptr);
    arma::fmat input_data = "1,2,5;"
                            "3,4,6;"
                            "7,8,9";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    for (int i = 0; i < 50; i++) {
        inputs.push_back(input);
    }
    upSample_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 50);
#ifdef DEBUG
    outputs[0]->show();
#endif
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        ASSERT_EQ(output->rows(), 6);
        ASSERT_EQ(output->cols(), 6);
        ASSERT_EQ(output->at(0, 0, 0), 1);
        ASSERT_EQ(output->at(0, 0, 2), 2);
        ASSERT_EQ(output->at(0, 2, 1), 3);
        ASSERT_EQ(output->at(0, 2, 3), 4);
        ASSERT_EQ(output->at(0, 3, 1), 3);
        ASSERT_EQ(output->at(0, 4, 1), 7);
    }
}

TEST(test_layer, upsample_Layer4) {
    using namespace kuiper_infer;
    std::shared_ptr<Operator> upSample_op = std::make_shared<UpSampleOperator>(1.5, 1.5, UpSampleMode::kModelNearest);
    std::shared_ptr<Layer> upSample_layer = LayerRegisterer::CreateLayer(upSample_op);
    CHECK(upSample_layer != nullptr);
    arma::fmat input_data = "1,2,5;"
                            "3,4,6;"
                            "7,8,9";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    inputs.push_back(input);
    inputs.push_back(input);
    upSample_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 3);
#ifdef DEBUG
    outputs[0]->show();
#endif
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        ASSERT_EQ(output->rows(), 5);
        ASSERT_EQ(output->cols(), 5);
        ASSERT_EQ(output->at(0, 0, 0), 1);
        ASSERT_EQ(output->at(0, 0, 2), 2);
        ASSERT_EQ(output->at(0, 2, 1), 3);
        ASSERT_EQ(output->at(0, 2, 3), 4);
        ASSERT_EQ(output->at(0, 3, 1), 3);
        ASSERT_EQ(output->at(0, 4, 1), 7);
    }
}