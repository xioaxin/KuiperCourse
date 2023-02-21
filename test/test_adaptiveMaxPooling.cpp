//
// Created by zpx on 2023/02/21.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/ops.h"
#include <cstdint>
#include "ops/adaptiveMaxPooling_op.h"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_adptiveMaxPooling1) {
    using namespace kuiper_infer;
    uint32_t output_h = 3;
    uint32_t output_w = 3;
    std::shared_ptr<Operator> adaptive_max_op = std::make_shared<AdaptiveMaxPoolingOperator>(output_h, output_w);
    std::shared_ptr<Layer> adaptive_max_layer = LayerRegisterer::CreateLayer(adaptive_max_op);
    CHECK(adaptive_max_layer != nullptr);
    arma::fmat input_data = "1 2 6 2 2 2 3 3 3;"
                            "1 3 1 2 8 2 4 9 4;"
                            "1 4 7 2 8 2 5 5 5;"
                            "1 5 1 2 21 2 6 6 22;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
    adaptive_max_layer->Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    const auto &output = outputs.at(0);
    ASSERT_EQ(output->rows(), 3);
    ASSERT_EQ(output->cols(), 3);
    ASSERT_EQ(output->at(0, 0, 0), 6);
    ASSERT_EQ(output->at(0, 0, 1), 8);
    ASSERT_EQ(output->at(0, 1, 0), 7);
    ASSERT_EQ(output->at(0, 1, 1), 8);
    ASSERT_EQ(output->at(0, 2, 1), 21);
    ASSERT_EQ(output->at(0, 2, 2), 22);
}
