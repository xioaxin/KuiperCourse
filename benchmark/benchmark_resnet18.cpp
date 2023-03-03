//
// Created by zpx on 2023/03/03.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "runtime/runtime_ir.h"
#include <benchmark/benchmark.h>

void benchmark_resnet18(const uint32_t batch_size) {
    using namespace kuiper_infer;
    const std::string &param_path = "../tmp/resnet18_hub.pnnx.param";
    const std::string &weight_path = "../tmp/resnet18_hub.pnnx.bin";
    RuntimeGraph graph(param_path, weight_path);
    graph.build("pnnx_input_0", "pnnx_output_0");
    const auto &operators = graph.operators();
    LOG(INFO) << "operator size: " << operators.size();
    std::vector<sftensor> inputs(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
        inputs.at(i) = std::make_shared<ftensor>(3, 224, 224);
        inputs.at(i)->fill(1.f);
    }
    const std::vector<sftensor> &outputs = graph.forward(inputs, false);
}

static void benchMarkResnet18(benchmark::State &state) {
    for (auto _: state)
        benchmark_resnet18(state.range(0));
}

BENCHMARK(benchMarkResnet18)->Arg(16)->Arg(32);