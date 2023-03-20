//
// Created by zpx on 2023/03/18.
//
#include "factory/transform_factory.h"
#include "transform/normalize.h"
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>

TEST(test_transform, test_normalize1) {
    using namespace kuiper_infer;
    arma::fmat input_data = "1,2,5;"
                            "3,4,6;"
                            "7,8,9";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    std::vector<float> mean = {0.0};
    std::vector<float> std = {1.0};
    float max_pixel_value = 255;
    TransformBase *normalize = new Normalize(mean, std, max_pixel_value);
    sftensor output = normalize->forward(input);
    const uint32_t rows = input->rows();
    const uint32_t cols = input->cols();
    for (int i = 0; i < output->channels(); ++i) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                ASSERT_EQ(output->at(i, row, col),
                          ((input->at(i, row, col) - (mean[i] * max_pixel_value)) / float(std[i] * max_pixel_value)));
            }
        }
    }
}

TEST(test_transform, test_normalize2) {
    using namespace kuiper_infer;
    arma::fmat input_data = "1,2,5;"
                            "3,4,6;"
                            "7,8,9";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_data.n_rows, input_data.n_cols);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;
    std::vector<float> mean = {0.0, 0.0, 0.0};
    std::vector<float> std = {1.0, 1.0, 1.0};
    float max_pixel_value = 255;
    TransformBase *normalize = new Normalize(mean, std, max_pixel_value);
    sftensor output = normalize->forward(input);
    const uint32_t rows = input->rows();
    const uint32_t cols = input->cols();
    for (int i = 0; i < output->channels(); ++i) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                ASSERT_EQ(output->at(i, row, col),
                          ((input->at(i, row, col) - (mean[i] * max_pixel_value)) / float(std[i] * max_pixel_value)));
            }
        }
    }
}

TEST(test_transform, test_normalize3) {
    using namespace kuiper_infer;
    arma::fmat input_data = "1,2,5;"
                            "3,4,6;"
                            "7,8,9";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_data.n_rows, input_data.n_cols);
    input->at(1) = input_data;
    input->at(2) = input_data;
    std::vector<sftensor> inputs;
    inputs.push_back(input);
    std::vector<float> mean = {0.3, 0.4, 0.5};
    std::vector<float> std = {1.0, 1.0, 1.0};
    float max_pixel_value = 255;
    std::vector<TransformBase *> v;
    v.push_back(new Normalize(mean, std, max_pixel_value));
    std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(3, input_data.n_rows, input_data.n_cols);
    std::vector<sftensor> outputs;
    outputs.push_back(output);
    TransformFactory *transformFactory = new TransformFactory(v);
    transformFactory->forward(inputs, outputs);
    const uint32_t rows = input->rows();
    const uint32_t cols = input->cols();
    for (int batch = 0; batch <outputs.size() ; ++batch) {
        output=outputs[batch];
        for (int i = 0; i < output->channels(); ++i) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    ASSERT_EQ(output->at(i, row, col),((input->at(i, row, col) - (mean[i] * max_pixel_value)) / float(std[i] * max_pixel_value)));
                }
            }
        }
    }

}