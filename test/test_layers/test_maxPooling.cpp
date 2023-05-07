//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/maxPooling_op.h"
#include "layer/layer.h"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_maxpooling1) {
    using namespace kuiper_infer;
    std::vector<int> stride = {1, 3};
    std::vector<int> padding_size = {0, 0};
    std::vector<int> kernel_size = {3, 3};
    std::vector<int> dilation = {3, 3};
    std::shared_ptr<RuntimeOperator> max_op = std::make_shared<MaxPoolingOperator>(kernel_size, padding_size, stride, dilation);
    std::shared_ptr<Layer> max_layer = LayerRegisterer::CreateLayer(max_op);
    CHECK(max_layer != nullptr);
    arma::fmat input_data = "71 22 63 94  65 16 75 58  9  11;"
                            "12 13 99 31 -31 55 99 857 12 511;"
                            "52 15 19 81 -61 15 49 67  12 41;"
                            "41 41 61 21 -15 15 10 13  51 55;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(MAX_TEST_ITERATION);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(MAX_TEST_ITERATION);
    for (int i = 0; i < MAX_TEST_ITERATION; i++) {
        inputs[i] = input;
    }
    max_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(0);
        ASSERT_EQ(output->rows(), 2);
        ASSERT_EQ(output->cols(), 3);
        ASSERT_EQ(output->at(0, 0, 0), 99);
        ASSERT_EQ(output->at(0, 0, 1), 94);
        ASSERT_EQ(output->at(0, 0, 2), 857);
        ASSERT_EQ(output->at(0, 1, 0), 99);
        ASSERT_EQ(output->at(0, 1, 1), 81);
        ASSERT_EQ(output->at(0, 1, 2), 857);
    }
}
