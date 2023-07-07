-- 加载C++库
local mylib = require("libluainfer")

-- 调用C++函数初始化SqueezeNet
mylib.initialize_squeezenet("squeezenet_v1.1.param", "squeezenet_v1.1.bin")

local startTime = os.clock() -- 记录开始时间

-- 调用C++函数进行图像推断
local scores = mylib.infer_squeezenet("zidane.jpg")

local endTime = os.clock() -- 记录结束时间

local elapsedTime = endTime - startTime -- 计算处理时间
print("函数处理时间: " .. elapsedTime .. "秒")

-- 输出前10个类别
print("top 10 cls:")
for i = 1, 10 do
    local score = scores[i]
    if score then
        print(i, "=", score)
    end
end


