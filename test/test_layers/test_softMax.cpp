//
// Created by zpx on 2022/12/27.
//
#include <gtest/gtest.h>
#include "ops/softMax_op.h"
#include "layer/softMax_layer.h"

TEST(test_layer, forward_softMax1) {
    using namespace kuiper_infer;
    std::shared_ptr<RuntimeOperator> softMax_op = std::make_shared<SoftMaxOperator>();
    arma::fmat input_data1 = "1,2,3,4;"
                             "5,6,7,8;"
                             "1,2,3,4;"
                             "5,6,7,8";
    arma::fmat input_data2 = "1,5,3,4;"
                             "7,2,3,4;"
                             "9,8,7,6;"
                             "5,4,3,2";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(2, 4, 4);
    input->at(0) = input_data1;
    input->at(1) = input_data2;
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(2, 4, 4);
    std::vector<sftensor> outputs(MAX_TEST_ITERATION);
    std::vector<sftensor> inputs(MAX_TEST_ITERATION); //作为一个批次去处理
    inputs[0] = input;
    for (int i = 0; i < MAX_TEST_ITERATION; i++) {
        inputs[i] = input;
    }
    SoftMaxLayer layer(softMax_op);
    layer.Forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), MAX_TEST_ITERATION);
    for (int i = 0; i < outputs.size(); i++) {
        output = outputs.at(0);
#ifdef DEBUG
        output->show();
#endif
        ASSERT_EQ(output->at(0, 0, 0), 0.5f);
        ASSERT_EQ(output->at(1, 0, 0), 0.5f);
    }
}
