//
// Created by zpx on 2023/03/18.
//

#ifndef KUIPER_COURSE_TRANSFORM_FACTORY_H
#define KUIPER_COURSE_TRANSFORM_FACTORY_H

#include <vector>
#include "data/tensor.hpp"
#include <opencv2/opencv.hpp>

namespace kuiper_infer {
    class TransformBase {
    public:
        TransformBase(std::string transform_name) : transform_name_(transform_name) {};

        TransformBase() {};
        virtual sftensor forward(sftensor &inputs) = 0;
        virtual ~TransformBase() = default;
    private:
        std::string transform_name_;
    };

    class TransformFactory {
    public:
        TransformFactory(std::vector<TransformBase*> &transformBase) : transforms_(transformBase) {};
        void forward(const std::vector<sftensor> &inputs, std::vector<sftensor> &outputs);

        ~TransformFactory() {};
    private:
        std::vector<TransformBase*> transforms_;
    };
}
#endif //KUIPER_COURSE_TRANSFORM_FACTORY_H
