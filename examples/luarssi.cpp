extern "C" {
    #include "lua.h"
    #include <lauxlib.h>
    #include <lualib.h>
}

#include "net.h"
#include <algorithm>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>

ncnn::Net model;

static void initialize_model(const std::string& param_path, const std::string& model_path, bool use_vulkan)
{
    model.opt.use_vulkan_compute = use_vulkan;

    if (model.load_param(param_path.c_str()))
    {
        std::cerr << "Failed to load param file: " << param_path << std::endl;
        exit(-1);
    }
    if (model.load_model(model_path.c_str()))
    {
        std::cerr << "Failed to load model file: " << model_path << std::endl;
        exit(-1);
    }
}

static void infer_model(std::vector<float>& input, std::vector<float>& cls_scores)
{
    ncnn::Mat in = ncnn::Mat(input.size(), 1, sizeof(float));
    memcpy(in.data, input.data(), input.size() * sizeof(float));

    ncnn::Extractor ex = model.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

    cls_scores.resize(out.w);

    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
        // std::cout << j << " " << out[j] << std::endl;
    }
}

// Lua functions
static int lua_initialize_model(lua_State* L)
{
    const char* param_path = lua_tostring(L, 1);
    const char* model_path = lua_tostring(L, 2);
    bool use_vulkan = lua_toboolean(L, 3);
    initialize_model(param_path, model_path, use_vulkan);
    return 0;
}

// Lua绑定函数
static int lua_infer_model(lua_State* L)
{
    // 检查参数
    if (!lua_istable(L, 1))
    {
        std::cerr << "Error: Invalid argument. Expected a table." << std::endl;
        return 0;
    }

    // 获取表的长度
    int tableLength = luaL_len(L, 1);
    // std::cout<<"tableLength:"<<tableLength<<std::endl;
    std::vector<float> input;

    // 遍历表并打印每个元素
    for (int i = 0; i <= tableLength; ++i)
    {
        // 获取表中的键值对
        lua_pushinteger(L, i);
        lua_gettable(L, 1);

        // 打印键
        // std::cout << "Key: " << lua_tostring(L, -2) << ", ";

        // 打印值
        // std::cout << "Value: " << lua_tonumber(L, -1) << std::endl;
        input.push_back(lua_tonumber(L, -1));

        // 弹出值和键，以便下一次迭代
        lua_pop(L, 1);
    }

    std::vector<float> cls_scores;
    infer_model(input, cls_scores);

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
static const luaL_Reg libluarssi[] = {
    {"initialize_model", lua_initialize_model},
    {"infer_model", lua_infer_model},
    {"print_topk", lua_print_topk},
    {NULL, NULL}
};

// 绑定库到全局命名空间
extern "C" int luaopen_libluarssi(lua_State* L)
{
    luaL_newlib(L, libluarssi);
    return 1;
}
