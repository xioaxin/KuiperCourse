//
// Created by zpx on 2023/02/21.
//

#ifndef KUIPER_COURSE_ADAPTIVEAVGPOOLING_H
#define KUIPER_COURSE_ADAPTIVEAVGPOOLING_H

#include "layer/layer.h"
#include <cstdint>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class AdaptiveAvgPoolingOperator : public RuntimeOperator {
    public:
        AdaptiveAvgPoolingOperator();

        ~AdaptiveAvgPoolingOperator() {};
        explicit AdaptiveAvgPoolingOperator(std::vector<int> output_size);
        const std::vector<int> &getOutputSize() const;
        void setOutputSize(const std::vector<int> &outputSize);
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(std::string type);
    private:
        std::vector<int> output_size_;
    };
}
#endif //KUIPER_COURSE_ADAPTIVEAVGPOOLING_H
