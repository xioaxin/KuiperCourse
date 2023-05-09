//
// Created by zpx on 2023/02/25.
//

#ifndef KUIPER_COURSE_CAT_OP_H
#define KUIPER_COURSE_CAT_OP_H

#include <cstdint>
#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class CatOperator : public RuntimeOperator {
    public:
        CatOperator();
        ~CatOperator() {};
        explicit CatOperator(uint32_t dim);
        void setDim(const uint32_t dim);
        const uint32_t getDim() const;
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
        uint32_t dim_;
    };
}
#endif //KUIPER_COURSE_CAT_OP_H
