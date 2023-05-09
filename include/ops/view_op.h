//
// Created by zpx on 2023/05/07.
//

#ifndef KUIPER_COURSE_VIEW_OP_H
#define KUIPER_COURSE_VIEW_OP_H

#include "runtime_op.h"
#include "factory/operator_factory.h"

namespace kuiper_infer {
    class ViewOperator : public RuntimeOperator {
    public:
        ViewOperator();
        explicit ViewOperator(std::vector<int> shape);

        ~ViewOperator() override {};
        void initialParameter(const std::map<std::string, std::shared_ptr<RuntimeParameter>> &runtimeParameter) override;
        void initialAttribute(const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &runtimeAttribute) override;
        static std::shared_ptr<RuntimeOperator> CreateInstance(const std::string type);
        void setShape(std::vector<int> shape);
        [[nodiscard]] std::vector<int> getShape() const;
    private:
        std::vector<int> shape_;
    };
}
#endif //KUIPER_COURSE_VIEW_OP_H
