//
// Created by zpx on 2023/04/01.
//
#include "ops/runtime_op.h"

namespace kuiper_infer {
    RuntimeOperator::RuntimeOperator() : op_type_(OpType::kOperatorUnknown) {}

    RuntimeOperator::~RuntimeOperator() {
    }

    RuntimeOperator::RuntimeOperator(OpType opType) : op_type_(opType) {}
}