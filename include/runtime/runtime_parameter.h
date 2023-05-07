//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_PARAMETER_H
#define KUIPER_COURSE_RUNTIME_PARAMETER_H

#include <string>
#include <status_code.h>
#include <vector>

namespace kuiper_infer {
    struct RuntimeParameter {
        virtual ~RuntimeParameter() = default;

        explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown) : type_(type) {}

        RuntimeParameterType type_ = RuntimeParameterType::kParameterUnknown;
    };

    struct RuntimeParameterInt : public RuntimeParameter {
        RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt) {}

        ~RuntimeParameterInt() = default;
        int value = 0;
    };

    struct RuntimeParameterFloat : public RuntimeParameter {
        RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}

        ~RuntimeParameterFloat() = default;
        float value = 0.f;
    };

    struct RuntimeParameterString : public RuntimeParameter {
        RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {}

        ~RuntimeParameterString() = default;
        std::string value;
    };

    struct RuntimeParameterIntArray : public RuntimeParameter {
        RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}

        ~RuntimeParameterIntArray() = default;
        std::vector<int> value;
    };

    struct RuntimeParameterFloatArray : public RuntimeParameter {
        RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}

        ~RuntimeParameterFloatArray() = default;
        std::vector<float> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter {
        RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}

        ~RuntimeParameterStringArray() = default;
        std::vector<std::string> value;
    };

    struct RuntimeParameterBool : public RuntimeParameter {
        RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {}

        ~RuntimeParameterBool() = default;
        bool value = false;
    };
}
#endif //KUIPER_COURSE_RUNTIME_PARAMETER_H
