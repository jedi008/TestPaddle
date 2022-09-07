from model import *
import gym, os
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

device = paddle.get_device()
env = gym.make("CartPole-v0")  ### 或者 env = gym.make("CartPole-v0").unwrapped 开启无锁定环境训练

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

if __name__ == '__main__':
    actor_weights = './model/actor.pdparams'
    if os.path.exists(actor_weights):
        actor = Actor(state_size, action_size)
        model_state_dict  = paddle.load(actor_weights)
        actor.set_state_dict(model_state_dict )
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size)
        
    
    state = env.reset()
    #初始话环境
    for t in range(1000):
        env.render()
        #提供环境

        state = paddle.to_tensor(state,dtype="float32",place=device)
        dist = actor(state)

        action = paddle.argmax(dist.logits)

        #在可行的动作空间中随机选择一个
        state, reward, done, info = env.step(action.cpu().squeeze(0).numpy())
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break