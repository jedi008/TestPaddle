import paddle
import paddle.nn as nn
from paddle.distribution import Normal
import numpy as np
from collections import deque
import random

# 定义评论家网络结构
# DDPG这种方法与Q学习紧密相关，可以看作是连续动作空间的深度Q学习。 
class Critic(nn.Layer):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256 + 1, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = self.relu(self.fc1(x))
        x = paddle.concat((x, a), axis=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义演员网络结构
# 为了使DDPG策略更好地进行探索，在训练时对其行为增加了干扰。 原始DDPG论文的作者建议使用时间相关的 OU噪声 ，
# 但最近的结果表明，不相关的均值零高斯噪声效果很好。 由于后者更简单，因此是首选。
class Actor(nn.Layer):
    def __init__(self, is_train=True):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.noisy = Normal(0, 0.2)
        self.is_train = is_train

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

    def select_action(self, epsilon, state):
        state = paddle.to_tensor(state,dtype="float32").unsqueeze(0)
        with paddle.no_grad():
            action = self.forward(state).squeeze() + self.is_train * epsilon * self.noisy.sample([1]).squeeze(0)
        return 2 * paddle.clip(action, -1, 1).numpy()

# 重播缓冲区:这是智能体以前的经验， 为了使算法具有稳定的行为，重播缓冲区应该足够大以包含广泛的体验。
# 如果仅使用最新数据，则可能会过分拟合，如果使用过多的经验，则可能会减慢模型的学习速度。 这可能需要一些调整才能正确。 
class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

