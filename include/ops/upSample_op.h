//
// Created by zpx on 2023/03/01.
//

#ifndef KUIPER_COURSE_UPSAMPLE_OP_H
#define KUIPER_COURSE_UPSAMPLE_OP_H

#include <cstdint>
#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    enum class UpSampleMode {
        kModelNearest = 0,
        kModelLinear = 1
    };

    class UpSampleOperator : public RuntimeOperator {
    public:
        UpSampleOperator();
        explicit UpSampleOperator(const float scale_h, const float scale_w,
                                  const UpSampleMode upSampleMode = UpSampleMode::kModelNearest);

        ~UpSampleOperator() {};
        const float getScaleH() const;
        void setScaleH(const float scale_h);
        const float getScaleW() const;
        void setScaleW(const float scale_w);
        void setUpSampleModel(const UpSampleMode upSampleMode);
        const UpSampleMode getUpSampleModel() const;
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        float scale_h_;
        float scale_w_;
        UpSampleMode upSampleMode1_ = UpSampleMode::kModelNearest;
    };
}
#endif //KUIPER_COURSE_UPSAMPLE_OP_H
