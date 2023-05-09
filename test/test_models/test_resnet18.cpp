// DESC: 加载Resnet18的模型进行分类推理
// Created by zpx on 2023/03/17.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string.h>
#include "data/load_data.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "runtime/runtime_ir.h"
#include "factory/transform_factory.h"
#include "transform/normalize.h"
#include "ops/softMax_op.h"
#include "layer/softMax_layer.h"

TEST(test_model, test_resnet18) {
    using namespace kuiper_infer;
    using namespace cv;
    using namespace std;
    const std::string &param_path = "../tmp/resnet18_hub.pnnx.param";
    const std::string &bin_path = "../tmp/resnet18_hub.pnnx.bin";
    const std::string &file_path = "../tmp/1.jpg";
    // 创建计算图
    RuntimeGraph graph(param_path, bin_path);
    graph.init(); //初始化
    graph.build("pnnx_input_0", "pnnx_output_0");
    // 读入图片
    sftensor input = ImageDataLoader::LoadData(file_path);
    vector<sftensor> inputs;
    vector<sftensor> outputs;
    inputs.push_back(input);
    // 归一化
    std::vector<float> mean = {0.3, 0.4, 0.5};
    std::vector<float> std = {1.0, 1.0, 1.0};
    float max_pixel_value = 255;
    std::vector<TransformBase *> v;
    v.push_back(new Normalize(mean, std, max_pixel_value));
    TransformFactory *transformFactory = new TransformFactory(v);
    transformFactory->forward(inputs, inputs);
    outputs = graph.forward(inputs, false); // 推理
    SoftMaxLayer layer(std::make_shared<SoftMaxOperator>());
    layer.Forwards(outputs, outputs);
//    for (int i = 0; i < outputs.size(); ++i) {
//        const sftensor &output_tensor = outputs.at(i);
//        output_tensor->show();
//        assert(output_tensor->size() == 1 * 1000);
//        float max_prob = -1;
//        int max_index = -1;
//        for (int j = 0; j < output_tensor->size(); ++j) {
//            float prob = output_tensor->index(j);
//            if (max_prob <= prob) {
//                max_prob = prob;
//                max_index = j;
//            }
//        }
//        printf("class with max prob is %f index %d\n", max_prob, max_index);
//    }
}