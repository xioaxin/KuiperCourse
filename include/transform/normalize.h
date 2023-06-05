//
// Created by zpx on 2023/03/18.
//

#ifndef KUIPER_COURSE_NORMALIZE_H
#define KUIPER_COURSE_NORMALIZE_H

#include "factory/transform_factory.h"

namespace kuiper_infer {
    class Normalize : public TransformBase {
    public:
        Normalize(const std::vector<float> &mean,const std::vector<float> &std, const float max_pixel_value = 255.0);
        sftensor forward(const kuiper_infer::sftensor &inputs) override;
        ~Normalize() override = default;
    private:
        std::vector<float> mean_;
        std::vector<float> std_;
        float max_pixel_value_;
    };
}
#endif //KUIPER_COURSE_NORMALIZE_H
