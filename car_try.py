import gym
import numpy as np
import pickle

if __name__=='__main__':
    env = gym.make('MountainCar-v0') #環境変数env
    time=200#手数
    discrese_num=30#離散値振り分けの個数
    q_table =np.zeros((discrese_num,discrese_num,3))
    observation=env.reset()#初期環境
    best_total_reward=-200#表示用の評価値
    actions=[]#行動
    best_actions=[]#良かった行動



    def to_discrese(_observation):
        low=env.observation_space.low
        high=env.observation_space.high
        dx=(high-low)/discrese_num
        position=int((_observation[0]-low[0])/dx[0])
        speed=int((_observation[1]-low[1])/dx[1])
        
        return position,speed


    def update(_q_table,_action,_observation,_next_observation,_reward,_timestep):
        alpha=0.2
        gamma=0.99

        next_position,next_speed=to_discrese(_next_observation)
        next_value=max(_q_table[next_position][next_speed])

        position,speed=to_discrese(_observation)
        pre_value=_q_table[position][speed][_action]

        _q_table[position][speed][_action]=pre_value+alpha*(_reward+gamma*next_value-pre_value)

        return _q_table


    def my_action(_env,_q_table,_observation,_timestep):
        epsilon=0.005
        if np.random.uniform(0,1)>epsilon:
            position,speed=to_discrese(observation)
            _action=np.argmax(_q_table[position][speed])
        else:
            _action=np.random.choice([0,1,2])
        
        return _action

    
    for timestep in range(2000):
        total_reward= 0 #1エピソードの評価
        observation=env.reset()
        for tr in range(time):
           # env.render()#描画
            action=my_action(env,q_table,observation,timestep)
            actions.append(action)
            next_observation, reward, done, info = env.step(action)
            q_table=update(q_table,action,observation,next_observation,reward,timestep)
            total_reward+=reward
            observation=next_observation
        
            if done:
                if best_total_reward<total_reward:
                    best_total_reward=total_reward
                    best_actions=actions
                if (timestep+1)%100==0:
                    print("Episode finished after {} timesteps,best_value:{}".format(timestep+1,200+best_total_reward))
                if (timestep+1)%2000==0:
                    f=open('best_actions.pkl','wb')
                    pickle.dump(best_actions,f)
                
                break

#main()
        
