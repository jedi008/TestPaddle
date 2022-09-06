# import gym
# env = gym.make('CartPole-v0')
# #生成环境
# env.reset()
# #重置环境，让环境回到起点
# for _ in range(100):
#     env.render()
#     #提供环境（把游戏中发生的显示到屏幕上）
#     env.step(env.action_space.sample()) 
#     # env.action_space.sample() 会在动作空间中随机选择一个
#     # env.step 会顺着这个动作进入下一个状态
# env.close()


import gym
env = gym.make('Pendulum-v1') #CartPole-v1
print("\naction_space: ",env.action_space)
print("\n\nobservation_space: ", env.observation_space)

 
'''
Discrete(2)
Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
'''
for i_episode in range(2):
    observation = env.reset()
    #初始话环境
    for t in range(1000):
        env.render()
        #提供环境
        action = env.action_space.sample()
        #在可行的动作空间中随机选择一个
        observation, reward, done, info = env.step(action)
        #顺着这个动作进入下一个状态
        print(observation, reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
 
'''
[-0.02344513 -0.17659043  0.0043245   0.27116755] 1.0 False {}
[-0.02697694 -0.37177384  0.00974786  0.5652113 ] 1.0 False {}
[-0.03441242 -0.56703115  0.02105208  0.8609492 ] 1.0 False {}
[-0.04575304 -0.7624334   0.03827107  1.1601765 ] 1.0 False {}
[-0.06100171 -0.9580325   0.0614746   1.4646091 ] 1.0 False {}
[-0.08016236 -1.1538512   0.09076678  1.7758446 ] 1.0 False {}
[-0.10323939 -0.9598616   0.12628368  1.5127068 ] 1.0 False {}
[-0.12243662 -1.1562663   0.1565378   1.8419967 ] 1.0 False {}
[-0.14556195 -0.9631801   0.19337773  1.6017431 ] 1.0 False {}
[-0.16482554 -0.7708017   0.2254126   1.3750535 ] 1.0 True {}
Episode finished after 10 timesteps
'''