//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_OPERAND_H
#define KUIPER_COURSE_RUNTIME_OPERAND_H

#include <vector>
#include <string>
#include <memory>
#include "status_code.h"
#include "runtime_datatype.h"
#include "data/tensor.hpp"

namespace kuiper_infer {
    struct RuntimeOperand {
        std::string name;
        std::vector<int32_t> shapes;
        std::vector<std::shared_ptr<Tensor<float>>> datas;
        RuntimeDataType type = RuntimeDataType::KTypeUnknown;
    };
}
#endif //KUIPER_COURSE_RUNTIME_OPERAND_H
