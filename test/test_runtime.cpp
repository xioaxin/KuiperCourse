//
// Created by zpx on 2023/02/12.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.h"

TEST(test_runtime, runtime1) {
    using namespace kuiper_infer;
    const std::string &param_path = "../tmp/test.pnnx.param";
    const std::string &bin_path = "../tmp/test.pnnx.bin";
    RuntimeGraph graph(param_path, bin_path);
    graph.init();
    const auto operators = graph.operators();
    for (const auto &operator_: operators) {
        LOG(INFO) << "Type: " << operator_->type << " name: " << operator_->name;
    }
}