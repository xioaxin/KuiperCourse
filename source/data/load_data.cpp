//
// Created by fss on 22-12-19.
//
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "data/load_data.hpp"
#include <glog/logging.h>
#include <regex>

namespace kuiper_infer {
    std::shared_ptr<Tensor<float >> CSVDataLoader::LoadDataWithHeader(const std::string &file_path,
                                                                      std::vector<std::string> &headers,
                                                                      char split_char) {
        CHECK(!file_path.empty()) << "File path is empty!";
        std::ifstream in(file_path);
        CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;
        std::string line_str;
        std::stringstream line_stream;
        const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
        CHECK(rows >= 1);
        std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float >>(1, rows - 1, cols);
        arma::fmat &data = input_tensor->at(0);
        size_t row = 0;
        while (in.good()) {
            std::getline(in, line_str);
            if (line_str.empty()) {
                break;
            }
            std::string token;
            line_stream.clear();
            line_stream.str(line_str);
            size_t col = 0;
            while (line_stream.good()) {
                std::getline(line_stream, token, split_char);
                try {
                    if (row == 0) {
                        if (token.substr(token.size() - 1, 2) == "\r")
                            headers.push_back(token.substr(0, token.size() - 1));
                        else headers.push_back(token);
                    } else data.at(row - 1, col) = std::stof(token);
                }
                catch (std::exception &e) {
                    LOG(ERROR) << "Parse CSV File meet error: " << e.what();
                    continue;
                }
                col += 1;
                CHECK(col <= cols) << "There are excessive elements on the column";
            }
            row += 1;
            CHECK(row <= rows) << "There are excessive elements on the row";
        }
        return input_tensor;
    }

    std::shared_ptr<Tensor<float >> CSVDataLoader::LoadData(const std::string &file_path, char split_char) {
        CHECK(!file_path.empty()) << "File path is empty!";
        std::ifstream in(file_path);
        CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;
        std::string line_str;
        std::stringstream line_stream;
        const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
        std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float >>(1, rows, cols);
        arma::fmat &data = input_tensor->at(0);
        size_t row = 0;
        while (in.good()) {
            std::getline(in, line_str);
            if (line_str.empty()) {
                break;
            }
            std::string token;
            line_stream.clear();
            line_stream.str(line_str);
            size_t col = 0;
            while (line_stream.good()) {
                std::getline(line_stream, token, split_char);
                try {
                    data.at(row, col) = std::stof(token);
                }
                catch (std::exception &e) {
                    LOG(ERROR) << "Parse CSV File meet error: " << e.what();
                    continue;
                }
                col += 1;
                CHECK(col <= cols) << "There are excessive elements on the column";
            }
            row += 1;
            CHECK(row <= rows) << "There are excessive elements on the row";
        }
        return input_tensor;
    }

    std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream &file, char split_char) {
        bool load_ok = file.good();
        file.clear();
        size_t fn_rows = 0;
        size_t fn_cols = 0;
        const std::ifstream::pos_type start_pos = file.tellg();
        std::string token;
        std::string line_str;
        std::stringstream line_stream;
        while (file.good() && load_ok) {
            std::getline(file, line_str);
            if (line_str.empty()) {
                break;
            }
            line_stream.clear();
            line_stream.str(line_str);
            size_t line_cols = 0;
            std::string row_token;
            while (line_stream.good()) {
                std::getline(line_stream, row_token, split_char);
                ++line_cols;
            }
            if (line_cols > fn_cols) {
                fn_cols = line_cols;
            }
            ++fn_rows;
        }
        file.clear();
        file.seekg(start_pos);
        return {fn_rows, fn_cols};
    }

    std::shared_ptr<Tensor<float>> ImageDataLoader::LoadData(const std::string &file_path) {
        CHECK(!file_path.empty()) << "The imag path is empty";
        struct stat buffer;
        CHECK((stat(file_path.c_str(), &buffer) == 0)) << "The path of image is illegal";
        cv::Mat image = cv::imread(file_path,cv::IMREAD_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3);
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float >>(image.channels(), image.rows, image.cols);
        uint8_t index = 0;
        uint32_t total_size_per_slice = image.rows * image.cols;
        for (const auto &item: channels) {
            const cv::Mat &split_image_t = item.t();
            memcpy(input_tensor->at(index).memptr(), split_image_t.data, sizeof(float) * total_size_per_slice);
            ++index;
        }
        return input_tensor;
    }
}