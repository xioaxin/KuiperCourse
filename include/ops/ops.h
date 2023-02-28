//
// Created by zpx on 2022/12/26.
//

#ifndef KUIPER_COURSE_OPS_H
#define KUIPER_COURSE_OPS_H
namespace kuiper_infer {
    enum class OpType {
        kOperatorUnknown = -1,
        kOperatorRelu = 0,
        kOperatorSigmoid = 1,
        kOperatorRelu6 = 2,
        kOperatorLeakyRelu = 3,
        kOperatorMaxPooling = 4,
        kOperatorExpression = 5,
        kOperatorAvgPooling = 6,
        kOperatorAdaptiveAvgPooling = 7,
        kOperatorAdaptiveMaxPooling = 8,
        kOperatorConvolution = 9,
        kOperatorBatchNorm = 10,
        kOperatorSoftMax = 11,
        kOperatorCat = 12,
        kOperatorFlatten = 13,
        kOperatorHardSigmoid = 14,
        kOperatorHardSwish = 15,
    };

    class Operator {
    public:
        OpType op_type_ = OpType::kOperatorUnknown;
        virtual ~Operator() = default;
        explicit Operator(OpType opType);
    };
}
#endif //KUIPER_COURSE_OPS_H
