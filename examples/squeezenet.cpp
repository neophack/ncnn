// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <iostream>

static void initialize_squeezenet(ncnn::Net& squeezenet, const std::string& param_path, const std::string& model_path)
{
    squeezenet.opt.use_vulkan_compute = true;

    if (squeezenet.load_param(param_path.c_str()))
    {
        std::cerr << "Failed to load param file: " << param_path << std::endl;
        exit(-1);
    }
    if (squeezenet.load_model(model_path.c_str()))
    {
        std::cerr << "Failed to load model file: " << model_path << std::endl;
        exit(-1);
    }
}

static void infer_squeezenet(const cv::Mat& bgr, const ncnn::Net& squeezenet, std::vector<float>& cls_scores)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    // Initialize SqueezeNet
    ncnn::Net squeezenet;
    std::string param_path = "squeezenet_v1.1.param";
    std::string model_path = "squeezenet_v1.1.bin";
    initialize_squeezenet(squeezenet, param_path, model_path);

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    // Perform inference
    std::vector<float> cls_scores;
    infer_squeezenet(m, squeezenet, cls_scores);

    print_topk(cls_scores, 3);

    return 0;
}
