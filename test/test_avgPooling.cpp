//
// Created by zpx on 2023/02/03.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/ops.h"
#include "ops/avgPooling_op.h"
#include "layer/layer.h"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_avgpooling1) {
    using namespace kuiper_infer;
    uint32_t stride_h = 1;
    uint32_t stride_w = 3;
    uint32_t padding_h = 0;
    uint32_t padding_w = 0;
    uint32_t pooling_h = 3;
    uint32_t pooling_w = 3;
    std::shared_ptr<Operator> avg_op = std::make_shared<AvgPoolingOperator>
            (pooling_h, pooling_w, stride_h, stride_w, padding_h, padding_w);
    std::shared_ptr<Layer> avg_layer = LayerRegisterer::CreateLayer(avg_op);
    CHECK(avg_layer != nullptr);
    arma::fmat input_data = "1 1 1 2 2 2 3 3 3;"
                            "1 1 1 2 2 2 4 4 4;"
                            "1 1 1 2 2 2 5 5 5;"
                            "1 1 1 2 2 2 6 6 6;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);
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
