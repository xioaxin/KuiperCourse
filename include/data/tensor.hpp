//
// Created by fss on 22-12-16. 基于fcube的数据类型包装成Tensor类
//

#ifndef KUIPER_COURSE_INCLUDE_TENSOR_HPP_
#define KUIPER_COURSE_INCLUDE_TENSOR_HPP_

#include <memory>
#include <vector>
#include <armadillo>

namespace kuiper_infer {
    template<typename T>
    class Tensor {
    };

    template<>
    class Tensor<uint8_t> {
        // 待实现，量化一个张量
    };

    template<>
    class Tensor<float> {
        // 元素都是float
    public:
        explicit Tensor() = default;
        explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols); // 通道、行数、列数
        explicit Tensor(const std::vector<uint32_t> &shape);
        static std::shared_ptr<Tensor<float>> create(uint32_t channels, uint32_t rows, uint32_t cols);
        Tensor(const Tensor &tensor);
        Tensor(Tensor &&tensor) noexcept;
        Tensor<float> &operator=(Tensor &&tensor) noexcept;
        Tensor<float> &operator=(const Tensor &tensor);
        uint32_t rows() const;
        uint32_t cols() const;
        uint32_t channels() const;
        uint32_t size() const;
        void set_data(const arma::fcube &data);
        bool empty() const;
        float index(uint32_t offset) const;
        float &index(uint32_t offset);
        std::vector<uint32_t> shapes() const;
        const std::vector<uint32_t> &raw_shapes() const;
        arma::fcube &data();
        const arma::fcube &data() const;
        arma::fmat &at(uint32_t channel);
        const arma::fmat &at(uint32_t channel) const;
        float at(uint32_t channel, uint32_t row, uint32_t col) const;
        float &at(uint32_t channel, uint32_t row, uint32_t col);
        void padding(const std::vector<uint32_t> &pads, float padding_value);
        void fill(float value);
        void fill(const std::vector<float> &values);
        void ones();
        void rand();
        void show();
        void flatten();
        void reRawShape(const std::vector<uint32_t> &shapes);
        void reRawView(const std::vector<uint32_t> &shapes);
        // 逐元素操作
        static std::shared_ptr<Tensor<float>> elementAdd(const std::shared_ptr<Tensor<float>> &tensor1,
                                                         const std::shared_ptr<Tensor<float>> &tensor2);
        static std::shared_ptr<Tensor<float>> elementMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                                                              const std::shared_ptr<Tensor<float>> &tensor2);
        static std::shared_ptr<Tensor<float>> elementSub(const std::shared_ptr<Tensor<float>> &tensor1,
                                                              const std::shared_ptr<Tensor<float>> &tensor2);
        static std::shared_ptr<Tensor<float>> elementDiv(const std::shared_ptr<Tensor<float>> &tensor1,
                                                              const std::shared_ptr<Tensor<float>> &tensor2);
        void transform(const std::function<float(float)> &filter);
        std::shared_ptr<Tensor<float>> clone();
        const float *raw_ptr() const;
    private:
        void reView(const std::vector<uint32_t> &shape);
        std::vector<uint32_t> raw_shapes_;
        arma::fcube data_;
    };
    using ftensor = Tensor<float>;
    using sftensor = std::shared_ptr<Tensor<float>>;
}
#endif //KUIPER_COURSE_INCLUDE_TENSOR_HPP_
