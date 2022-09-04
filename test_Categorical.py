import paddle
from paddle.distribution import Categorical
import paddle.nn.functional as F


paddle.seed(100) # on CPU device
x = F.softmax(paddle.rand([6]), axis=-1)
print(x)
# [0.5535528  0.20714243 0.01162981
#  0.51577556 0.36369765 0.2609165 ]

cat = Categorical(x)
# 作用是创建以参数probs为标准的类别分布，样本是来自 “0 … K-1” 的整数，其中 K 是probs参数的长度。也就是说，按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。

for i in range(1):
    action = cat.sample([1]) # 在cat按照各个位置的概率随机选取其中一个位置，返回位置Index
    print("action: ", action)

value = paddle.to_tensor([2,1,3])
print("cat.prob(value): ", cat.prob(value))
print("cat.log_prob(value): ", cat.log_prob(value)) # 底数是e,对各个对应位置的值进行log计算

