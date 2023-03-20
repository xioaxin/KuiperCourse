//
// Created by fss on 22-12-19.
// 加载CSV数据：
//  1.直接读取CSV正文数据，按照分割符号分割然后读取入Tensor中
//  2.读取CSV数据标题和正文数据

#ifndef KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_

#include <armadillo>
#include "tensor.hpp"
#include <sys/stat.h>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/core.hpp>

namespace kuiper_infer {
    class CSVDataLoader {
    public:
        static std::shared_ptr<Tensor<float >> LoadData(const std::string &file_path, char split_char = ',');
        static std::shared_ptr<Tensor<float >> LoadDataWithHeader(const std::string &file_path,
                                                                  std::vector<std::string> &headers,
                                                                  char split_char = ',');

        ~CSVDataLoader() {};
    private:
        static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
    };

//    class ImageDataLoader { // 读取图像数据
//    public:
//        static std::shared_ptr<Tensor<float >> LoadData(const std::string &file_path);
//
//        ~ImageDataLoader() {};
//    };
}
#endif //KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
