//
// Created by zpx on 2022/12/26.
//

#ifndef KUIPER_COURSE_LAYER_H
#define KUIPER_COURSE_LAYER_H

#include "data/tensor.hpp"
#include <string>

namespace kuiper_infer {
    class Layer {
    public:
        explicit Layer(const std::string &layer_name);
        virtual void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs);
        virtual ~Layer() = default;
    private:
        std::string layer_name_;
    };
}
#endif //KUIPER_COURSE_LAYER_H
