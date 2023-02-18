//
// Created by fss on 22-12-19.
// 加载CSV数据：
//  1.直接读取CSV正文数据，按照分割符号分割然后读取入Tensor中
//  2.读取CSV数据标题和正文数据

#ifndef KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#include <armadillo>
#include "data/tensor.hpp"

namespace kuiper_infer {
class CSVDataLoader {
 public:
  static std::shared_ptr<Tensor<float >> LoadData(const std::string &file_path, char split_char = ',');

  static std::shared_ptr<Tensor<float >> LoadDataWithHeader(const std::string &file_path,
                                       std::vector<std::string> &headers, char split_char = ',');

 private:
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
};
}
#endif //KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
