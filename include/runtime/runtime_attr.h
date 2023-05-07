//
// Created by zpx on 2023/02/11.
//

#ifndef KUIPER_COURSE_RUNTIME_ATTR_H
#define KUIPER_COURSE_RUNTIME_ATTR_H

#include <vector>
#include <glog/logging.h>
#include "runtime_datatype.h"
#include <algorithm>

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
        std::vector<std::shared_ptr<Tensor<T>>> get();

        static std::vector<sftensor> get_value(std::shared_ptr<RuntimeAttribute> runtimeAttribute) {
            return runtimeAttribute->get<float>();
        };

        static std::vector<sftensor> get_value_vector(std::shared_ptr<RuntimeAttribute> runtimeAttribute, uint32_t batch_sizes) {
            return runtimeAttribute->copy_data_vector<float>({1, static_cast<unsigned int>(runtimeAttribute->shape[0]), 1}, batch_sizes);
        };

        static std::vector<sftensor> get_value_matrix(std::shared_ptr<RuntimeAttribute> runtimeAttribute, uint32_t batch_sizes) {
            return runtimeAttribute->copy_data_vector<float>({1,static_cast<unsigned int>(runtimeAttribute->shape[0]),
                                                              static_cast<unsigned int>(runtimeAttribute->shape[1])}, batch_sizes);
        };
        template<typename T>
        std::vector<std::shared_ptr<Tensor<T>>> copy_data(std::vector<uint32_t> shape);
        template<typename T>
        std::vector<std::shared_ptr<Tensor<T>>> copy_data_vector(std::vector<uint32_t> shape, uint32_t batch_sizes);
    };

    template<class T>
    std::vector<std::shared_ptr<Tensor<T>>> RuntimeAttribute::get() {
        // 检查节点属性中的权重类型
        CHECK(!weight_data.empty());
        CHECK(type != RuntimeDataType::KTypeUnknown);
        std::vector<std::shared_ptr<Tensor<T>>> weights;
        switch (shape.size()) {
            case 1:
                weights = copy_data<T>({1, 1, 1});
                break;
            case 4:
                weights = copy_data<T>(
                        {static_cast<unsigned int>(shape[1]), static_cast<unsigned int>(shape[2]), static_cast<unsigned int>(shape[3])});
                break;
            default:
                LOG(FATAL) << "shape size error";
        }
        return weights;
    }

    template<typename T>
    std::vector<std::shared_ptr<Tensor<T>>> RuntimeAttribute::copy_data(const std::vector<uint32_t> shape_) {
        uint32_t size = sizeof(T);
        for (auto item: shape_) {
            if (item == 0)break;
            size *= item;
        }
        T *data = (T *) malloc(size);
        std::shared_ptr<ftensor> data_ = std::make_shared<ftensor>(shape_);
        uint32_t batch_sizes = this->shape[0];
        if (this->shape.size() == 2)batch_sizes = 1;
        std::vector<std::shared_ptr<Tensor<T>>> weights(batch_sizes);
#pragma omp parallel for num_threads(batch_sizes)
        for (int batch_size = 0; batch_size < batch_sizes; ++batch_size) {
            memcpy(data, (float *) weight_data.data() + batch_size * size, size);
            data_->fill(data, false);
            weights[batch_size] = data_;
        }
        return weights;
    }

    template<typename T>
    std::vector<std::shared_ptr<Tensor<T>>> RuntimeAttribute::copy_data_vector(const std::vector<uint32_t> shape_, uint32_t batch_sizes) {
        uint32_t size = sizeof(T);
        for (auto item: shape_) {
            if (item == 0)break;
            size *= item;
        }
        T *data = (T *) malloc(size);
        std::shared_ptr<ftensor> data_ = std::make_shared<ftensor>(shape_);
        std::vector<std::shared_ptr<Tensor<T>>> weights(batch_sizes);
#pragma omp parallel for num_threads(batch_sizes)
        for (int batch_size = 0; batch_size < batch_sizes; ++batch_size) {
            memcpy(data, (float *) weight_data.data() + batch_size * size, size);
            data_->fill(data, false);
            weights[batch_size] = data_;
        }
        return weights;
    }
}
#endif //KUIPER_COURSE_RUNTIME_ATTR_H
