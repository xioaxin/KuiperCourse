//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/runtime_op.h"
#include "ops/avgPooling_op.h"

TEST(test_layer, forward_avgpooling1) {
    using namespace kuiper_infer;
    std::vector<int> stride = {1, 3};
    std::vector<int> padding_size = {0, 0};
    std::vector<int> kernel_size = {3, 3};
    std::vector<int> dilation = {3, 3};
    std::shared_ptr<RuntimeOperator> avg_op = std::make_shared<AvgPoolingOperator>(kernel_size, padding_size, stride, dilation);
    std::shared_ptr<Layer> avg_layer = LayerRegisterer::CreateLayer(avg_op);
    CHECK(avg_layer != nullptr);
    arma::fmat input_data = "1 1 1 2 2 2 3 3 3;"
                            "1 1 1 2 2 2 4 4 4;"
                            "1 1 1 2 2 2 5 5 5;"
                            "1 1 1 2 2 2 6 6 6;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(1);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
    inputs[0] = input;
    avg_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    const auto &output = outputs.at(0);
    ASSERT_EQ(output->rows(), 2);
    ASSERT_EQ(output->cols(), 3);
    ASSERT_EQ(output->at(0, 0, 0), 1);
    ASSERT_EQ(output->at(0, 0, 1), 2);
    ASSERT_EQ(output->at(0, 0, 2), 4);
    ASSERT_EQ(output->at(0, 1, 0), 1);
    ASSERT_EQ(output->at(0, 1, 1), 2);
    ASSERT_EQ(output->at(0, 1, 2), 5);
}
TEST(test_layer, forward_avgpooling2) {
    using namespace kuiper_infer;
    std::vector<int> stride = {1, 3};
    std::vector<int> padding_size = {0, 0};
    std::vector<int> kernel_size = {3, 3};
    std::vector<int> dilation = {3, 3};
    std::shared_ptr<RuntimeOperator> avg_op = std::make_shared<AvgPoolingOperator>(kernel_size, padding_size, stride, dilation);
    std::shared_ptr<Layer> avg_layer = LayerRegisterer::CreateLayer(avg_op);
    CHECK(avg_layer != nullptr);
    arma::fmat input_data = "1 1 1 2 2 2 3 3 3;"
                            "1 1 1 2 2 2 4 4 4;"
                            "1 1 1 2 2 2 5 5 5;"
                            "1 1 1 2 2 2 6 6 6;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(20);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(20);
    for (int i = 0; i < 20; ++i) {
        inputs[i] = input;
    }
    avg_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 20);
    for (auto &output: outputs) {
        ASSERT_EQ(output->rows(), 2);
        ASSERT_EQ(output->cols(), 3);
        ASSERT_EQ(output->at(0, 0, 0), 1);
        ASSERT_EQ(output->at(0, 0, 1), 2);
        ASSERT_EQ(output->at(0, 0, 2), 4);
        ASSERT_EQ(output->at(0, 1, 0), 1);
        ASSERT_EQ(output->at(0, 1, 1), 2);
        ASSERT_EQ(output->at(0, 1, 2), 5);
    }
}