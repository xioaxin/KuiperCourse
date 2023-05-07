//
// Created by zpx on 2023/02/26.
//

#ifndef KUIPER_COURSE_FLATTEN_H
#define KUIPER_COURSE_FLATTEN_H

#include <cstdint>
#include <map>
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class FlattenOperator : public RuntimeOperator {
    public:
        FlattenOperator();
        explicit FlattenOperator(int start_dim, int end_dim);

        ~FlattenOperator() {};
        void initialParameter(const std::map<std::string, RuntimeParameter *> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
    private:
    public:
        int getStartDim() const;
        void setStartDim(int startDim);
        int getEndDim() const;
        void setEndDim(int endDim);
    private:
        int start_dim_;
        int end_dim_;
    };
}
#endif //KUIPER_COURSE_FLATTEN_H
