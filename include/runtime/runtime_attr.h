//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_ATTR_H
#define KUIPER_COURSE_RUNTIME_ATTR_H

#include <vector>
#include <glog/logging.h>
#include "runtime_datatype.h"

namespace kuiper_infer {
/// 计算图节点的属性信息
    struct RuntimeAttribute {
        std::vector<char> weight_data; /// 节点中的权重参数
        std::vector<int> shape;  /// 节点中的形状信息
        RuntimeDataType type = RuntimeDataType::KTypeUnknown; /// 节点中的数据类型

        /**
         * 从节点中加载权重参数
         * @tparam T 权重类型
         * @return 权重参数数组
         */
        template<class T>
        //
        std::vector<T> get();
    };

    template<class T>
    std::vector<T> RuntimeAttribute::get() {
        /// 检查节点属性中的权重类型
        CHECK(!weight_data.empty());
        CHECK(type != RuntimeDataType::KTypeUnknown);
        std::vector<T> weights;
        switch (type) {
            case RuntimeDataType::KTypeFloat32: { /// 加载的数据类型是float
                const bool is_float = std::is_same<T, float>::value;
                CHECK_EQ(is_float, true);
                const uint32_t float_size = sizeof(float);
                CHECK_EQ(weight_data.size() % float_size, 0);
                for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
                    float weight = *((float *) weight_data.data() + i);
                    weights.push_back(weight);
                }
                break;
            }
            default: {
                LOG(FATAL) << "Unknown weight data type";
            }
        }
        return weights;
    }
}
#endif //KUIPER_COURSE_RUNTIME_ATTR_H
