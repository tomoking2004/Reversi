# coding: utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.losses import huber_loss
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

from config import *
from game import SimpleReversi
from agents import Random_Agent


class DQN_Agent:

    def __init__(self, env):
        self.env = env
        self.mainQN = self.build_model()
        self.targetQN = self.build_model()
        self.memory = deque(maxlen=MEMORY_SIZE)

    def build_model(self):  # ネットワーク構築
        model = Sequential()
        # when loading model, faild with InputLayer.
        model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape=INPUT_SHAPE))
        model.add(BatchNormalization())
        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(OUTPUT_SIZE, activation='softmax'))
        optimizer = RMSprop(learning_rate=LR)
        # when loading model, faild with huber loss function.
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def info(self):  # 情報表示
        self.mainQN.summary()
        plot_model(self.mainQN, to_file="models/{}_network.png".format(DQN_MODEL_NAME), show_shapes=True)

    def remember(self, experience):  # 経験記憶
        self.memory.append(experience)

    def replay(self, batch_size):  # 経験再生

        if len(self.memory) <= batch_size:
            return
        
        inputs = np.zeros((batch_size, *INPUT_SHAPE))
        targets = np.zeros((batch_size, OUTPUT_SIZE))
        mini_batch = random.sample(self.memory, batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all():
                retmainQs = self.mainQN.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + GAMMA * self.targetQN.predict(next_state_b)[0][next_action]

            targets[i] = self.mainQN.predict(state_b)
            targets[i][action_b] = target

        self.mainQN.fit(inputs, targets, epochs=1, verbose=0)

    def update_target(self):  # target更新
        self.targetQN.set_weights(self.mainQN.get_weights())

    def get_action(self, epsilon=0.0):  # 行動

        state = self.env.get_state()

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = self.mainQN.predict(state)[0]
            action = np.argmax(retTargetQs)
        else:
            action = np.random.choice(self.env.actions)

        return action

    def load(self, path):  # 反映
        try:
            self.mainQN = load_model(path)
            print('Loaded DQN model.')
        except:
            print('Failed loading DQN model.')

    def save(self, path):  # 保存
        try:
            self.mainQN.save(path)
            print('Saved DQN model.')
        except:
            print('Failed saving DQN model.')

    def train(self):  # 学習

        # メインルーチン
        total_reward = np.array([])
        total_mean = np.array([])
        islearned = False

        self.info()

        for episode in range(1, NUM_EPISODES+1):

            state = self.env.reset()
            self.update_target()
            episode_reward = 0

            for t in range(1, MAX_STEPS+1):

                epsilon = 0.001 + 0.9 / (1.0+episode)
                action = self.get_action(epsilon)

                next_state, reward, done, _ = self.env.step(action)
                self.remember((state, action, reward, next_state))
                state = next_state
                episode_reward += reward

                if not islearned:
                    self.replay(BATCH_SIZE)

                if DQN_MODE:
                    self.update_target()

                if done:
                    break

            # 報酬を記録
            total_reward = np.hstack((total_reward, episode_reward))
            mean = total_reward[-AVG_SIZE:].mean()
            total_mean = np.hstack((total_mean, mean))
            print('Episode %d, finished after %2d time steps | reward %2d | mean %.1f' % (episode, t, episode_reward, mean))

            # 終了を判断
            if mean >= GOAL_AVG_REWARD and episode >= AVG_SIZE:
                print('Episode %d, trained successfuly!' % episode)
                islearned = True

            # モデルの定期セーブ
            if episode%SAVE_CYCLE==0:
                self.save('models/{}_{}.h5'.format(DQN_MODEL_NAME, episode))

        # モデルのセーブ
        self.save('models/{}_{}.h5'.format(DQN_MODEL_NAME, episode))

        # 結果のプロット
        plt.plot(np.arange(NUM_EPISODES), total_mean)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("result")
        plt.savefig("models/{}_result.png".format(DQN_MODEL_NAME))
        plt.show()


if __name__=="__main__":
    
    env = SimpleReversi()
    dqn = DQN_Agent(env)
    #dqn.load(None)
    dqn.train()
