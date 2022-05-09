import paddle.fluid as fluid

# 定义两个张量
x1 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
x2 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
print("x1: ",x1)
print("x2: ",x2)

# 将两个张量求和
y1 = fluid.layers.sum(x=[x1, x2])

print("y1: ",y1)


# 可能是copy到了老版本的代码，以下代码报错
# # 创建一个使用CPU的执行器
# place = fluid.CPUPlace()
# exe = fluid.executor.Executor(place)
# # 进行参数初始化
# exe.run(fluid.default_startup_program())

# # 进行运算，并把y的结果输出
# result = exe.run(program=fluid.default_main_program(),
#                  fetch_list=[y1])
# print(result)