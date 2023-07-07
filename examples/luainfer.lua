-- 加载C++库
local mylib = require("libluainfer")

-- 调用C++函数初始化SqueezeNet
mylib.initialize_squeezenet("squeezenet_v1.1.param", "squeezenet_v1.1.bin")

-- 调用C++函数进行图像推断
local scores = mylib.infer_squeezenet("zidane.jpg")

-- 输出前10个类别
print("top 10 cls:")
for i = 1, 10 do
    local score = scores[i]
    if score then
        print(i, "=", score)
    end
end

print("top k res:")
-- 打印排序后前k个结果
local topk = 5
mylib.print_topk(scores, topk)

