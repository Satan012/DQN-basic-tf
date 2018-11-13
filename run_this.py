from DQN_modified import DeepQNetwork
from maze_env import Maze
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def run_maze():
    sucesses = []
    step = 0  # 用来控制什么时候学习
    for episode in range(300):
        # 初始化环境
        observation = env.reset()
        sucess_count = 0


        while True:
            # 刷新环境
            env.render()

            # DQN 根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, done = env.step(action)
            if done:
                sucess_count += 1
                print(sucess_count)

            # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)  # 将用于经验回放

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            if step >200 and step % 100 == 0:
                print('sucess_count:', sucess_count)
                sucesses.append(sucess_count)

            # 将下一个 state_ 变为 下次循环的 state
            observation = observation_

            # 如果终止, 就跳出循环
            if done:
                break
            step += 1  # 总步数

    # end of game
    print('game over')
    env.destroy()

    plt.plot(np.arange(len(sucesses)), sucesses)
    plt.ylabel('Count')
    plt.xlabel('training steps')
    plt.show()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=2000,  # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )
    env.after(100, run_maze)
    env.mainloop()
    # RL.plot_cost()  # 观看神经网络的误差曲线
