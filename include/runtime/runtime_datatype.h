//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_DATATYPE_H
#define KUIPER_COURSE_RUNTIME_DATATYPE_H
enum class RuntimeDataType {
    KTypeUnknown = 0,
    KTypeFloat32 = 1,
    KTypeFloat64 = 2,
    KTypeFloat16 = 3,
    KTypeInt32 = 4,
    KTypeInt64 = 5,
    KTypeInt16 = 6,
    KTypeInt8 = 7,
    KTypeUInt8 = 8,
};
#endif //KUIPER_COURSE_RUNTIME_DATATYPE_H
