import gym
import paddle
import paddle.nn as nn
from itertools import count
from paddle.distribution import Normal
import numpy as np
from collections import deque
import random
import paddle.nn.functional as F
from visualdl import LogWriter


from model import * 

# 定义软更新的函数
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.set_value( target_param * (1.0 - tau) + param * tau) 
       
# 定义环境、实例化模型
env = gym.make('Pendulum-v1')
actor = Actor()
critic = Critic()
actor_target = Actor()
critic_target = Critic()

# 定义优化器
critic_optim = paddle.optimizer.Adam(parameters=critic.parameters(), learning_rate=3e-5)
actor_optim = paddle.optimizer.Adam(parameters=actor.parameters(), learning_rate=1e-5)

# 定义超参数
explore = 50000
epsilon = 1
gamma = 0.99
tau = 0.001

memory_replay = Memory(50000)
begin_train = False
batch_size = 32

learn_steps = 0
epochs = 250

writer = LogWriter('logs')

# 训练循环
for epoch in range(0, epochs):
    state = env.reset()
    episode_reward = 0
    for time_step in range(200):
        action = actor.select_action(epsilon, state)
        next_state, reward, done, _ = env.step([action])
        episode_reward += reward
        reward = (reward + 8.1) / 8.1
        memory_replay.add((state, next_state, action, reward))
        if memory_replay.size() > 1280:

            learn_steps += 1
            if not begin_train:
                print('train begin!')
                begin_train = True
            experiences = memory_replay.sample(batch_size, False)
            batch_state, batch_next_state, batch_action, batch_reward = zip(*experiences)

            batch_state = paddle.to_tensor(batch_state,dtype="float32")
            batch_next_state = paddle.to_tensor(batch_next_state,dtype="float32")
            batch_action = paddle.to_tensor(batch_action,dtype="float32").unsqueeze(1)
            batch_reward = paddle.to_tensor(batch_reward,dtype="float32").unsqueeze(1)


            # 均方误差 y - Q(s, a) ， y是目标网络所看到的预期收益， 而 Q(s, a)是Critic网络预测的操作值。
            # y是一个移动的目标，评论者模型试图实现的目标；这个目标通过缓慢的更新目标模型来保持稳定。 
            with paddle.no_grad():
                Q_next = critic_target(batch_next_state, actor_target(batch_next_state))
                Q_target = batch_reward + gamma * Q_next

            critic_loss = F.mse_loss(critic(batch_state, batch_action), Q_target)


            critic_optim.clear_grad()
            critic_loss.backward()
            critic_optim.step()

            writer.add_scalar('critic loss', critic_loss.numpy(), learn_steps)
            # 使用Critic网络给定值的平均值来评价Actor网络采取的行动。 我们力求使这一数值最大化。 
            # 因此，我们更新了Actor网络，对于一个给定状态，它产生的动作尽量让Critic网络给出高的评分。 
            critic.eval()
            actor_loss = - critic(batch_state, actor(batch_state))
            # print(actor_loss.shape)
            actor_loss = actor_loss.mean()
            actor_optim.clear_grad()
            actor_loss.backward()
            actor_optim.step()
            critic.train()
            writer.add_scalar('actor loss', actor_loss.numpy(), learn_steps)

            soft_update(actor_target, actor, tau)  
            soft_update(critic_target, critic, tau) 


        if epsilon > 0:
            epsilon -= 1 / explore
        state = next_state

    writer.add_scalar('episode reward', episode_reward, epoch)
    if epoch % 50 == 0:
        print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
    
    if epoch % 200 == 0:
        paddle.save(actor.state_dict(), 'model/ddpg-actor' + str(epoch) + '.para')
        paddle.save(critic.state_dict(), 'model/ddpg-critic' + str(epoch) + '.para')
        print('model saved!')  
