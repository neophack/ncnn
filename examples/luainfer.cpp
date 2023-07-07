#include <lua5.4/lua.h>
#include <lua5.4/lauxlib.h>
#include <lua5.4/lualib.h>

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

ncnn::Net squeezenet;

static void initialize_squeezenet(const std::string& param_path, const std::string& model_path)
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

static void infer_squeezenet(const std::string& imagepath, std::vector<float>& cls_scores)
{
    cv::Mat bgr = cv::imread(imagepath, 1);

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

// Lua functions
static int lua_initialize_squeezenet(lua_State* L)
{
    const char* param_path = lua_tostring(L, 1);
    const char* model_path = lua_tostring(L, 2);
    initialize_squeezenet(param_path, model_path);
    return 0;
}

// Lua绑定函数
static int lua_infer_squeezenet(lua_State* L)
{
    const char* imagepath = luaL_checkstring(L, 1); // 获取第一个参数
    std::vector<float> cls_scores;

    infer_squeezenet(imagepath, cls_scores);

    // 将结果压入Lua栈
    lua_newtable(L);
    for (size_t i = 0; i < cls_scores.size(); i++)
    {
        // std::cerr << "i: " << i <<" "<<cls_scores[i]<< std::endl;
        lua_pushnumber(L, cls_scores[i]);
        lua_rawseti(L, -2, i + 1);
    }
    // lua_settable(L,-cls_scores.size()-1);
    return 1; // 返回结果个数
}

static int lua_print_topk(lua_State* L)
{
    // Get the input table from Lua
    luaL_checktype(L, 1, LUA_TTABLE);
    int topk = luaL_checkinteger(L, 2);

    // Get the size of the input table
    int size = lua_rawlen(L, 1);

    // Create a vector to store the scores and indices
    std::vector<std::pair<float, int> > vec(size);

    // Populate the vector with the scores and indices from the input table
    for (int i = 0; i < size; i++)
    {
        lua_rawgeti(L, 1, i + 1);
        float score = static_cast<float>(luaL_checknumber(L, -1));
        lua_pop(L, 1);
        vec[i] = std::make_pair(score, i);
    }

    // Partially sort the vector based on the scores in descending order
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // Print the topk scores and indices
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

// 注册函数到Lua库
static const luaL_Reg libluainfer[] = {
    {"initialize_squeezenet", lua_initialize_squeezenet},
    {"infer_squeezenet", lua_infer_squeezenet},
    {"print_topk", lua_print_topk},
    {NULL, NULL}
};

// 绑定库到全局命名空间
extern "C" int luaopen_libluainfer(lua_State* L)
{
    luaL_newlib(L, libluainfer);
    return 1;
}
