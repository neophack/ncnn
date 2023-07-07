-- 加载C++库
local perception = require("libluarssi")

-- 调用C++函数初始化SqueezeNet
perception.initialize_model("huawei_p50.param", "huawei_p50.bin",True)

local tableLength = 120
local loops = 10
local myTable = {}

for i = 1, tableLength do
    myTable[i] = math.random(-100, 0)
end

-- 打印表的内容
-- for i = 1, tableLength do
--     print(myTable[i])
-- end

local startTime = os.clock() -- 记录开始时间
local scores;
for i = 1, loops do
    -- 调用C++函数进行推断
    scores = perception.infer_model(myTable)
end

local endTime = os.clock() -- 记录结束时间

local elapsedTime = endTime - startTime -- 计算处理时间
print(loops .. "次循环，推理时间: " .. elapsedTime .. "秒")

-- 输出前10个类别
print("res:")
for i = 1, 10 do
    local score = scores[i]
    if score then
        print(i, "=", score)
    end
end
